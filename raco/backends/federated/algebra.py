import os
import time
from raco import algebra
from raco import rules
from raco.relation_key import RelationKey
from raco.backends import Language, Algebra
from raco.backends.myria import MyriaLeftDeepTreeAlgebra as MyriaAlgebra
from raco.compile import optimize
from raco.viz import operator_to_dot
from raco.backends.myria import compile_to_json
from raco.backends.scidb import SciDBAFLAlgebra
import raco.algebra
import itertools
import subprocess

from raco.backends.myria.catalog import MyriaCatalog
from raco.backends.scidb.catalog import SciDBCatalog
from raco.backends.spark.catalog import SparkCatalog
from raco.representation import RepresentationProperties

class Federated(Language):
    pass


class FederatedOperator(algebra.ZeroaryOperator):
    language = Federated

    def scheme(self):
        raise NotImplementedError

    def shortStr(self):
        return repr(self)

    def partitioning(self):
        return RepresentationProperties()


class FederatedExec(FederatedOperator):
    '''Execute a plan on a specific backend'''
    def __init__(self, plan, catalog):
        # Logical plan to be optimized and executed on target platform
        # Catalog is needed for optimization and identifies target
        self.plan = plan
        self.catalog = catalog

    def scheme(self):
        return self.plan.scheme()

    def num_tuples(self):
        return self.plan.num_tuples()

    def __str__(self):
        return "{}Exec({})".format(self.catalog.__class__.__name__, self.plan)

class FederatedMove(FederatedOperator):
    def __init__(self, sourcename, sourcecatalog, targetname, targetcatalog):
        self.sourcename = sourcename
        self.sourcecatalog = sourcecatalog
        self.targetname = targetname
        self.targetcatalog = targetcatalog

    def num_tuples(self):
        return 0

    def __str__(self):
        args = (self.sourcename, self.targetcatalog.__class__.__name__, self.targetname)
        return "Move({} TO {} AS {})".format(*args)


class FederatedSequence(raco.algebra.Sequence, FederatedOperator):
    def __init__(self, args):
        for expr in args:
            assert(isinstance(expr, FederatedOperator))
        super(self.__class__, self).__init__(args)
        # The last operator in the sequence provides the return value
        assert(isinstance(args[-1], FederatedExec))

    @property
    def plan(self):
        return self.args[-1].plan

    @plan.setter
    def plan(self, plan):
        self.args[-1].plan = plan

    @property
    def catalog(self):
        return self.args[-1].catalog

    @catalog.setter
    def catalog(self, catalog):
        self.args[-1].catalog = catalog

    def __str__(self):
        args = ",\n\n".join([str(x) for x in self.args])
        return "{}(\n{}\n)".format(self.__class__.__name__, args)


class FederatedParallel(raco.algebra.Parallel, FederatedOperator):
    def __init__(self, args):
        for expr in args:
            assert(isinstance(expr, FederatedOperator))
        super(self.__class__, self).__init__(args)


class FederatedDoWhile(raco.algebra.DoWhile, FederatedOperator):
    def __init__(self, args):
        for expr in args:
            assert(isinstance(expr, FederatedOperator))
        super(self.__class__, self).__init__(args)

class FederatedSingletonRelation(raco.algebra.SingletonRelation, FederatedOperator):
    def __init__(self):
        self.plan = []

class Runner(FederatedExec):
    pass

class RunAQL(Runner):
    """Run an AQL query on a SciDB instance specified by the programmer"""

    def __repr__(self):
        return "RunAQL(%s, %s)" % (self.command, self.connection)

    def num_tuples(self):
      raise NotImplementedError("{op}.num_tuples".format(op=type(self)))


class RunMyria(Runner):
    """Run a Myria query on the UW cluster"""

    def __repr__(self):
        return "RunMyria(%s, %s)" % (self.command, self.connection)

    def num_tuples(self):
      raise NotImplementedError("{op}.num_tuples".format(op=type(self)))

#class ExportMyriaToScidb(Mover):
#    def __init__(self, myria_relkey, scidb_array_name, conn):
#       self.scidb_array_name = scidb_array_name
#        self.myria_relkey = myria_relkey
#        self.connection = conn

#    def shortStr(self):
#        return "ExportToScidb(%s, %s)" % (self.myria_relkey,
                                          #self.scidb_array_name)

#    def copy(self, other):
#        self.scidb_array_name = other.scidb_array_name
#        self.myria_relkey = other.myria_relkey
#        self.connection = other.connection
#
#    def scheme(self):
#        raise NotImplementedError()


dispatchmap = {"aql": RunAQL, "myria": RunMyria, "afl": RunAQL}


class Dispatch(rules.Rule):

    def fire(self, expr):
        if isinstance(expr, FederatedSequence):
            return expr

        if isinstance(expr, FederatedMove):
            if isinstance(expr.sourcecatalog, SciDBCatalog) \
                              and isinstance(expr.targetcatalog, MyriaCatalog):
                # Construct parallel load plan for Myria
                loadplan = expr.plan
                return RunMyria(loadplan, expr.targetcatalog)
            else:
                raise NotImplementedError()

        if isinstance(expr, algebra.ExecScan):
            # Some kind of custom code that we must pass through
            return dispatchmap[expr.languagetag](expr.command, expr.connection)

        if isinstance(expr, FederatedExec):
            if isinstance(expr.catalog, MyriaCatalog):
                pp = optimize(expr, target=MyriaAlgebra())
                json = compile_to_json("raw query", "logical plan", pp)
                return RunMyria(json)

            if isinstance(expr.catalog, SciDBCatalog):
                pp = optimize(expr, target=SciDBAFLAlgebra())
                aql = compile_to_aql(pp)
                return RunAQL(aql, expr.catalog)


class SplitBackend(rules.BottomUpRule):
    err = "Expected child op {} to be a federated plan.  \
Maybe rule traversal is not bottom-up?"

    def __init__(self, catalog):
        # Assumes this is a Federated Catalog
        self.federatedcatalog = catalog
        super(self.__class__, self).__init__()

    @classmethod
    def checkchild(cls, child):
        if not isinstance(child, FederatedOperator):
            raise ValueError(cls.err.format(child))

    def fire(self, op):
        # print type(op)
        if isinstance(op, raco.algebra.Scan):
            # TODO: Assumes each relation is in only one catalog
            cat = self.federatedcatalog.sourceof(op.relation_key)
            newop = FederatedExec(op, cat)
            return newop

        if isinstance(op, raco.algebra.ScanTemp):
            # TODO: Assumes each relation is in only one catalog
            cat = self.federatedcatalog.get_catalog(op.name)
            newop = FederatedExec(op, cat)
            return newop

        if isinstance(op, raco.algebra.EmptyRelation):
            # Assuming empty relations are scidb for now
            return FederatedExec(op, self.federatedcatalog.get_scidb_catalog())

        if isinstance(op, raco.algebra.UnaryOperator):
           self.checkchild(op.input)

           if isinstance(op, raco.algebra.StoreTemp):
               self.federatedcatalog.add_to_temp_relations(op.name, op.input.catalog)

           execop = op.input
           # Absorb the current operator into the Exec
           op.input = op.input.plan
           execop.plan = op

           return execop

        if isinstance(op, raco.algebra.BinaryOperator):
           self.checkchild(op.left)
           self.checkchild(op.right)

           leftcatalog = op.left.catalog
           rightcatalog = op.right.catalog

           if leftcatalog == rightcatalog:
               op.left = op.left.plan
               op.right = op.right.plan
               newexec = FederatedExec(op, leftcatalog)
               return newexec

           else:
               if isinstance(leftcatalog, MyriaCatalog) and \
                             isinstance(rightcatalog, SciDBCatalog):
                       # We need to move a dataset

                   # Give it a name
                   movedrelation = RelationKey(raco.algebra.gensym())

                   # Add a store operation on the SciDB side
                   scidbwork = op.right
                   scidbwork.plan = raco.algebra.Store(movedrelation, scidbwork.plan)


                   # Create the Move operator
                   mover = FederatedMove(movedrelation,
                                         rightcatalog,
                                         movedrelation,
                                         leftcatalog)

                   # Wrap the current operator on Myria
                   myriawork = op.left
                   op.left = op.left.plan
                   # insert a scan of the moved relation on the Myria side
                   op.right = raco.algebra.Scan(movedrelation)
                   myriawork.plan = op

                   # Create a Sequence operator to define execution order
                   federatedplan = FederatedSequence([scidbwork, mover, myriawork])

                   return federatedplan

               elif isinstance(rightcatalog, MyriaCatalog) and \
                             isinstance(leftcatalog, SciDBCatalog):
                   # We need to move a dataset; flipped repetition of above
                   # TODO: abstract this better

                   # Give it a name
                   movedrelation = RelationKey(raco.algebra.gensym())

                   # Add a store operation on the SciDB side
                   scidbwork = op.left
                   scidbwork.plan = raco.algebra.Store(movedrelation, scidbwork.plan)


                   # Create the Move operator
                   mover = FederatedMove(movedrelation,
                                         leftcatalog,
                                         movedrelation,
                                         rightcatalog)

                   # Wrap the current operator on Myria
                   myriawork = op.right
                   op.right = op.right.plan
                   # insert a scan of the moved relation on the Myria side
                   op.left = raco.algebra.Scan(movedrelation)
                   myriawork.plan = op

                   # Create a Sequence operator to define execution order
                   federatedplan = FederatedSequence([scidbwork, mover, myriawork])

                   return federatedplan

               else:
                   template = "Expected Myria or SciDB catalogs, got {}, {}"
                   msg = template.format(leftcatalog, rightcatalog)
                   raise NotImplementedError(msg)

        elif isinstance(op, raco.algebra.NaryOperator):
            # We have a hybrid plan
            if isinstance(op, raco.algebra.Sequence):
                return FederatedSequence(op.args)

            if isinstance(op, raco.algebra.Parallel):
                return FederatedParallel(op.args)

            if isinstance(op, raco.algebra.DoWhile):
                return FederatedDoWhile(op.args)


        assert False, "{op} --- is not supported".format(op = op)
        return op

class SplitSparkToMyria(rules.BottomUpRule):
    err = "Expected child op {} to be a federated plan.  \
Maybe rule traversal is not bottom-up?"

    def __init__(self, catalog):
        # Assumes this is a Federated Catalog
        self.federatedcatalog = catalog
        super(self.__class__, self).__init__()

    def __str__(self):
        return "SplitSparkToMyria"

    @classmethod
    def checkchild(cls, child):
        if not isinstance(child, FederatedOperator):
            raise ValueError(cls.err.format(child))

    def fire(self, op):
        # print type(op)
        if isinstance(op, raco.algebra.Scan):
            # TODO: Assumes each relation is in only one catalog
            cat = self.federatedcatalog.sourceof(op.relation_key)
            newop = FederatedExec(op, cat)
            return newop

        if isinstance(op, raco.algebra.ScanTemp):
            # TODO: Assumes each relation is in only one catalog
            cat = self.federatedcatalog.get_catalog(op.name)
            newop = FederatedExec(op, cat)
            return newop

        if isinstance(op, raco.algebra.EmptyRelation):
            # Assuming empty relations are spark for now
            return FederatedExec(op, self.federatedcatalog.get_spark_catalog())

        if isinstance(op, raco.algebra.SingletonRelation):
            # Assuming singleton relations are spark for now
            return FederatedExec(FederatedSingletonRelation(), self.federatedcatalog.get_spark_catalog())

        if isinstance(op, raco.algebra.UnaryOperator):
           self.checkchild(op.input)

           if isinstance(op, raco.algebra.StoreTemp):
               self.federatedcatalog.add_to_temp_relations(op.name, op.input.catalog)

           execop = op.input
           # Absorb the current operator into the Exec
           op.input = op.input.plan
           execop.plan = op

           return execop

        if isinstance(op, raco.algebra.BinaryOperator):
           self.checkchild(op.left)
           self.checkchild(op.right)

           leftcatalog = op.left.catalog
           rightcatalog = op.right.catalog

           if leftcatalog == rightcatalog:
               op.left = op.left.plan
               op.right = op.right.plan
               newexec = FederatedExec(op, leftcatalog)
               return newexec

           else:
               if isinstance(leftcatalog, MyriaCatalog) and \
                             isinstance(rightcatalog, SparkCatalog):
                       # We need to move a dataset

                   # Give it a name
                   movedrelation = RelationKey(raco.algebra.gensym())

                   # Add a store operation on the Myria side
                   myriawork = op.left
                   myriawork.plan = raco.algebra.FileStore(path=os.path.join(os.path.abspath(os.path.curdir), movedrelation.relation), format='CSV',  plan=myriawork.plan)


                   # Create the Move operator
                   mover = FederatedMove(movedrelation,
                                         leftcatalog,
                                         movedrelation,
                                         rightcatalog)

                   # Wrap the current operator on Spark
                   sparkwork = op.right
                   op.right = op.right.plan
                   # insert a scan of the moved relation on the Spark side
                   op.left = raco.algebra.Scan(movedrelation, myriawork.plan.scheme())
                   sparkwork.plan = op

                   # Create a Sequence operator to define execution order
                   federatedplan = FederatedSequence([myriawork, mover, sparkwork])
                       # Insert shell command here to run Accumulo and generate file
                   print myriawork.plan.__repr__()


                   subprocess.call('java -cp /home/dhutchis/gits/lara-graphulo/target/lara-graphulo-1.0-SNAPSHOT-all.jar edu.washington.cs.laragraphulo.Main \"{}\"'.format(myriawork.plan.__repr__()), shell=True)

                   while(True):
                       if os.path.exists(os.path.join(os.path.abspath(os.path.curdir), movedrelation.relation)):
                           break
                   time.sleep(2)
                   return federatedplan

               elif isinstance(rightcatalog, MyriaCatalog) and \
                             isinstance(leftcatalog, SparkCatalog):
                   # We need to move a dataset; flipped repetition of above
                   # TODO: abstract this better

                   # Give it a name
                   movedrelation = RelationKey(raco.algebra.gensym())

                   # Add a store operation on the Myria side
                   myriawork = op.right
                   myriawork.plan = raco.algebra.FileStore(path=os.path.join(os.path.abspath(os.path.curdir), movedrelation.relation), format='CSV',  plan=myriawork.plan)


                   # Create the Move operator
                   mover = FederatedMove(movedrelation,
                                         rightcatalog,
                                         movedrelation,
                                         leftcatalog)

                   # Wrap the current operator on Spark
                   sparkwork = op.left
                   op.left = op.left.plan
                   # insert a scan of the moved relation on the Spark side
                   op.right = raco.algebra.Scan(movedrelation, myriawork.plan.scheme())
                   sparkwork.plan = op

                   # Create a Sequence operator to define execution order
                   federatedplan = FederatedSequence([myriawork, mover, sparkwork])
                   # Insert shell command here to run Accumulo and generate file
                   print myriawork.plan.__repr__()


                   subprocess.call('java -cp /home/dhutchis/gits/lara-graphulo/target/lara-graphulo-1.0-SNAPSHOT-all.jar edu.washington.cs.laragraphulo.Main \"{}\"'.format(myriawork.plan.__repr__()), shell=True)


                   while(True):
                       if os.path.exists(os.path.join(os.path.abspath(os.path.curdir), movedrelation.relation)):
                           break
                       time.sleep(2)
                   return federatedplan

               else:
                   template = "Expected Myria or Spark catalogs, got {}, {}"
                   msg = template.format(leftcatalog, rightcatalog)
                   raise NotImplementedError(msg)

        elif isinstance(op, raco.algebra.NaryOperator):
            # We have a hybrid plan
            if isinstance(op, raco.algebra.Sequence):
                return FederatedSequence(op.args)

            if isinstance(op, raco.algebra.Parallel):
                return FederatedParallel(op.args)

            if isinstance(op, raco.algebra.DoWhile):
                return FederatedDoWhile(op.args)


        assert False, "{op} --- is not supported".format(op = op)
        return op


class FlattenSingletonFederatedSequence(rules.TopDownRule):
    def fire(self, op):
        if isinstance(op, FederatedSequence):
            if len(op.args) == 1:
                if isinstance(op.args[0], FederatedSequence):
                    return op.args[0]
        return op

class FederatedAlgebra(Algebra):
    language = Federated

    operators = [FederatedExec, FederatedMove, FederatedSequence]

    def __init__(self, algebras, catalog, crossproducts=True):
        '''
        A list of algebras and a federated catalog
        '''
        self.algebras = algebras
        self.federatedcatalog = catalog
        self.crossproducts = crossproducts

    def opt_rules(self, **kwargs):
        opt_logical_rules = [rules.RemoveTrivialSequences(),
                rules.SimpleGroupBy(),
                rules.SplitSelects(),
                rules.PushSelects(),
                rules.MergeSelects(),
                rules.ProjectToDistinctColumnSelect(),
                rules.JoinToProjectingJoin(),
                rules.PushApply(),
                rules.RemoveUnusedColumns(),
                rules.PushApply(),
                rules.RemoveUnusedColumns(),
                rules.PushApply()]
        if self.crossproducts:
            fedrules = [
                # opt_logical_rules,
                [rules.CrossProduct2Join()],
                rules.push_select,
                [SplitSparkToMyria(self.federatedcatalog)]]
                # [FlattenSingletonFederatedSequence()]]
                #Dispatch()]
        else:
            fedrules = [rules.push_select, [SplitSparkToMyria(self.federatedcatalog)]]
        return list(itertools.chain(*fedrules))
