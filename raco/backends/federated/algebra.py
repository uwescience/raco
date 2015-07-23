from raco import algebra
from raco import rules
from raco.backends import Language, Algebra
from raco.backends.myria import MyriaLeftDeepTreeAlgebra as MyriaAlgebra
from raco.compile import optimize
from raco.backends.myria import compile_to_json
import raco.algebra

from raco.backends.myria.catalog import MyriaCatalog
from raco.backends.scidb.catalog import SciDBCatalog

class Federated(Language):
    pass


class FederatedOperator(algebra.ZeroaryOperator):
    language = Federated

    def scheme(self):
        raise NotImplementedError

    def shortStr(self):
        return repr(self)

class FederatedExec(FederatedOperator):
    '''Execute a plan on a specific backend'''
    def __init__(self, plan, catalog):
        # Logical plan to be optimized and executed on target platform
        # Catalog is needed for optimization and identifies target
        self.plan = plan
        self.catalog = catalog

    def num_tuples(self):
        return self.plan.num_tuples()

    def __repr__(self):
        return "{}Exec({})".format(self.catalog.__class__.__name__, self.plan)

class FederatedMove(FederatedOperator):
    def __init__(self, sourcename, sourcecatalog, targetname, targetcatalog):
        self.sourcename = sourcename
        self.sourcecatalog = sourcecatalog
        self.targetname = targetname
        self.targetcatalog = targetcatalog

    def num_tuples(self):
        return 0

    def __repr__(self):
        args = (self.sourcename, self.targetcatalog.__class__.__name__, self.targetname)
        return "Move({} TO {} AS {})".format(*args)

class FederatedSequence(raco.algebra.Sequence, FederatedExec):
    def __init__(self, args):
        for expr in args:
            assert(isinstance(expr, FederatedOperator))
        super(self.__class__, self).__init__(args)
        # The last operator in the sequence provides the return value
        assert(isinstance(args[-1], FederatedExec))
        self.plan = args[-1].plan
        self.catalog = args[-1].catalog

class FederatedParallel(FederatedSequence):
    pass

#class RunAQL(Runner):
#    """Run an AQL query on a SciDB instance specified by the programmer"""
#    def __repr__(self):
#        return "RunAQL(%s, %s)" % (self.command, self.connection)

#    def num_tuples(self):
#      raise NotImplementedError("{op}.num_tuples".format(op=type(self)))


#class RunMyria(Runner):
#    """Run a Myria query on the UW cluster"""

#    def __repr__(self):
#        return "RunMyria(%s, %s)" % (self.command, self.connection)

#    def num_tuples(self):
#      raise NotImplementedError("{op}.num_tuples".format(op=type(self)))

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


#dispatchmap = {"aql": RunAQL, "myria": RunMyria, "afl": RunAQL}


class Dispatch(rules.Rule):
    def fire(self, expr):
        if isinstance(expr, algebra.Sequence):
            return expr  # Retain top-level sequence operator
        if isinstance(expr, algebra.Move):
            return expr
        if isinstance(expr, algebra.ExecScan):
            # Some kind of custom code that we must pass through
            return dispatchmap[expr.languagetag](expr.command, expr.connection)
        if isinstance(expr, FederatedExec):
            return dispatchmap[expr.languagetag](expr.command, expr.connection)
        else:
            # Just a logical plan that we will dispatch to Myria by default
            pp = optimize(expr, target=MyriaAlgebra())
            json = compile_to_json("raw query", "logical plan", pp)
            return dispatchmap["myria"](json)



'''
class ToSciDB(rules.Rule):
    pass
    # One strategy: start from datasets in SciDB, grow fragments until you hit
    # operators you don't want to do in SciDB
    #
class LoopUnroll(rules.Rule):
    def fire(self, op):
        if isinstance(op, algebra.DoWhile):

'''


class SplitSciDBToMyria(rules.BottomUpRule):
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
        if isinstance(op, raco.algebra.Scan):
            # TODO: Assumes each relation is in only one catalog
            cat = self.federatedcatalog.sourceof(op.relation_key)
            newop = FederatedExec(op, cat)
            return newop

        if isinstance(op, raco.algebra.UnaryOperator):
           self.checkchild(op.input)

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
                   movedrelation = raco.algebra.gensym()

                   # Add a store operation on the SciDB side
                   scidbwork = op.right
                   scidbwork.plan = raco.algebra.Store(scidbwork.plan, movedrelation)


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

               elif isinstance(leftcatalog, MyriaCatalog) and \
                             isinstance(rightcatalog, SciDBCatalog):
                   # We need to move a dataset; flipped repetition of above
                   # TODO: abstract this better

                   # Give it a name
                   movedrelation = raco.algebra.gensym()

                   # Add a store operation on the SciDB side
                   scidbwork = op.left
                   scidbwork.plan = raco.algebra.Store(scidbwork.plan, movedrelation)


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
                   raise NotImplemented(msg)

        if isinstance(op, raco.algebra.Sequence):
            return FederatedSequence(op.args)

        if isinstance(op, raco.algebra.NaryOperator):
            template = "NaryOperators not yet implemented, got {}"
            msg = template.format(op)
            raise NotImplementedError(msg)

class FederatedAlgebra(Algebra):
    language = Federated

    operators = [FederatedExec, FederatedMove]

    def __init__(self, algebras, catalog):
        '''
        A list of algebras and a federated catalog
        '''
        self.algebras = algebras
        self.federatedcatalog = catalog

    def opt_rules(self, **kwargs):
        childrules = sum([a.opt_rules() for a in self.algebras],[])
        fedrules = [SplitSciDBToMyria(self.federatedcatalog)]
        return fedrules
