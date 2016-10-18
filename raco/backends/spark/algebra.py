import logging
import itertools

from raco import rules
from raco.backends import Language, Algebra
from raco.backends.federated.algebra import FederatedExec, FederatedSequence, FederatedDoWhile
from raco import algebra
from raco.scheme import Scheme
from raco.expression import *

from copy import copy
class SparkLanguage(Language):
    pass


LOGGER = logging.getLogger(__name__)



class SparkOperator(object):
    pass

class SparkScan(algebra.Scan, SparkOperator):
    def compileme(self):
        return "scan({})".format(str(self.relation_key).split(':')[-1])

class SparkScanTemp(algebra.ScanTemp, SparkOperator):
    def compileme(self):
        return "scan({})".format(str(self.relation_key).split(':')[-1])

class SparkStore(algebra.Store, SparkOperator):
    def compileme(self, inputid):
        return "store({},{})".format(self.relation_key, inputid)

class SparkConcat(algebra.UnionAll, SparkOperator):
    def compileme(self, leftid, rightid):
        return "concat({},{})".format(leftid, rightid)

class SparkSelect(algebra.Select, SparkOperator):
    def compileme(self, input):
        return "filter({}, {})".format(input, remove_unnamed_literals(self, self.condition))
        # return "filter({}, {})".format(input, compile_expr(self.condition, self.scheme(), None))

class SparkJoin(algebra.Join, SparkOperator):
    def compileme(self, left, right):
        return "{}.join({}, {})".format(left, right, remove_unnamed_literals(self, self.condition))

class SparkProject(algebra.Project, SparkOperator):
    def compileme(self, input):
        return "select(input, {})".format(input, ", ".join(["'" + x.name + "'" for x in self.columnlist]))

class SparkAggregate(algebra.GroupBy, SparkOperator):
    def compileme(self, input):
        new_agg_list = list()
        for i, agg in enumerate(self.aggregate_list):
            new_agg_list.append(str(agg) + ' as _COLUMN%d_' % i) # Todo: Really pathetic hack, fix later.
        if len(self.grouping_list) == 0:
            return "aggregate({input}, {aggregate_list})"\
                .format(input = input,
                        aggregate_list = ",".join([x for x in new_agg_list]))
        return "aggregate({input}, {aggregate_list}, {dimension_list})"\
            .format(input=input,
                    aggregate_list = ",".join([x for x in new_agg_list]),
                    dimension_list = ",".join([str(dim) for dim in self.grouping_list]))

class SparkApply(algebra.Apply, SparkOperator):
    def compileme(self, input):
        return "apply({}, {})".format(input, ",".join([remove_unnamed_literals(self.input, str(x))
                                                       for x in list(itertools.chain(*self.emitters))]))

class SparkGroupBy(algebra.GroupBy, SparkOperator):
    def compileme(self, input):
        # Todo: fix later, doesn't matter right now. Till compile to scala is working.
        return "GroupBy"

class SparkOrderBy(algebra.OrderBy, SparkOperator):
    def compileme(self, input):
        # Todo: fix later, doesn't matter right now. Till compile to scala is working.
        return "OrderBy"

class SparkStoreTemp(algebra.StoreTemp, SparkOperator):
    def compileme(self, input):
        return "StoreTemp"

class SparkSequence(algebra.Sequence, SparkOperator):
    def compileme(self, input):
        return "Sequence"

class SparkDoWhile(algebra.DoWhile, SparkOperator):
    def compileme(self, input):
        return "DoWhile"

### RULES

class ApplyToApplyProject(rules.BottomUpRule):
    def fire(self, expr):
        if isinstance(expr, SparkApply):
            # Checking for just projection operation.
            just_project = True
            for (n, ex) in expr.emitters:
                if isinstance(ex, NamedAttributeRef):
                    if n != ex.name:
                        just_project = False
                else:
                    just_project = False

            input_to_project = expr.input if just_project else expr


            toreturn = SparkProject([NamedAttributeRef(name) for (name, expr_to_apply) in expr.emitters], input_to_project)
            return toreturn
        return expr

    def __str__(self):
        return "SciDBApply => SciDBApply followed by a SciDbProject"

class GroupByToGroupByApply(rules.Rule):
    def fire(self, expr):
        if isinstance(expr, algebra.GroupBy):
            for aggr in expr.aggregate_list:
                if not (isinstance(aggr, COUNTALL) or isinstance(aggr.input, NamedAttributeRef) or isinstance(aggr.input, UnnamedAttributeRef)):
                    # print type(aggr.input)
                    # The aggr is a function.
                    # Spark doesn't support this, so we will add an apply below groupby to evaluate the expr
                    prev_input = expr.input
                    new_col_name = aggr.__class__.__name__ + str(random.randint(1, 1000000))
                    print expr.grouping_list
                    gp_col_list = map(lambda col: (col.debug_info, col), expr.grouping_list)
                    new_input = SparkApply(gp_col_list + [(new_col_name, aggr.input)], prev_input)
                    aggr.input = NamedAttributeRef(new_col_name)
                    new_gp_list = []
                    for c in range(len(expr.grouping_list)):
                        new_gp_list.append(NamedAttributeRef(str(expr.grouping_list[c])))
                    expr.grouping_list = new_gp_list
                    expr.input = new_input
        return expr

class ScanTempToSparkScanTemp(rules.Rule):
    def fire(self, expr):
        if isinstance(expr, algebra.ScanTemp):
            expr_scheme = expr._scheme
            if len(expr.scheme().attributes) == 1:
                if expr.scheme().attributes[0][0] == "_COLUMN0_":
                    expr_scheme = Scheme([(expr.name, expr.scheme().attributes[0][1])])
            # print 'scheme for: ' , expr.name, expr_scheme
            sctemp = SparkScanTemp(expr.name, expr_scheme)
            # TODO: I should copy the expr onto sctemp here, but that causes the schema to be overwritten. Fix later.
            sctemp.name = expr.name
            return sctemp
        return expr

class ConsumeFederatedOps(rules.Rule):
    def fire(self, expr):
        if isinstance(expr, FederatedExec):
            return expr.plan
        if isinstance(expr, FederatedSequence):
            return algebra.Sequence(expr.args)
        if isinstance(expr, FederatedDoWhile):
            return algebra.DoWhile(expr.args)
        return expr
### ALGEBRA

class SparkAlgebra(Algebra):
    """ Spark algebra abstract class"""
    language = SparkLanguage

    operators = [
        SparkScan,
        SparkStore,
        SparkSelect,
        SparkApply,
        SparkProject,
        SparkJoin,
        SparkGroupBy,
        SparkOrderBy,
        SparkStoreTemp,
        SparkDoWhile,
        SparkSequence
    ]

    def opt_rules(self, **kwargs):
        # replace logical operator with its corresponding SciDB operators
        sparkify = [
            rules.OneToOne(algebra.Store, SparkStore),
            rules.OneToOne(algebra.Scan, SparkScan),
            rules.OneToOne(algebra.ScanTemp, SparkScanTemp),
            rules.OneToOne(algebra.Select, SparkSelect),
            rules.OneToOne(algebra.Apply, SparkApply),
            rules.OneToOne(algebra.Project, SparkProject),
            rules.OneToOne(algebra.Join, SparkJoin),
            rules.OneToOne(algebra.GroupBy, SparkGroupBy),
            rules.OneToOne(algebra.OrderBy, SparkOrderBy),
            rules.OneToOne(algebra.StoreTemp, SparkStoreTemp),
            rules.OneToOne(algebra.DoWhile, SparkDoWhile),
            rules.OneToOne(algebra.Sequence, SparkSequence)
        ]
        all_rules = [ConsumeFederatedOps()] + sparkify + [GroupByToGroupByApply(), ScanTempToSparkScanTemp()]

        return all_rules

    def __init__(self, catalog=None):
        self.catalog = catalog

def compile_to_scala(plan):
    pass

def compile_plan(plan):
    pass
    # if isinstance(plan, SciDBStore):
    #     return "\nstore(" + compile_plan(plan.input, scidb_out_relation, temp_out_name) + \
    #           ", {scidb_out_relation});".format(scidb_out_relation=str(plan.relation_key).replace(':', '__'))
    # if isinstance(plan, SciDBRegrid):
    #     return plan.compileme(compile_plan(plan.input, scidb_out_relation, temp_out_name))
    # if isinstance(plan, SciDBRedimension):
    #     # plan.template_array = temp_out_name
    #     return plan.compileme(compile_plan(plan.input, scidb_out_relation, temp_out_name))
    # if isinstance(plan, (SciDBSelect, SciDBProject, SciDBApply)):
    #     return plan.compileme(compile_plan(plan.input, scidb_out_relation, temp_out_name))
    # if isinstance(plan, SciDBScan):
    #     return plan.compileme()
    # if isinstance(plan, SciDBJoin):
    #     return plan.compileme(compile_plan(plan.left, scidb_out_relation, temp_out_name),
    #                           compile_plan(plan.right, scidb_out_relation, temp_out_name))
    # if isinstance(plan, SciDBAggregate):
    #     return plan.compileme(compile_plan(plan.input, scidb_out_relation, temp_out_name))
    # if isinstance(plan, SciDbMult):
    #     return plan.compileme(compile_plan(plan.left, scidb_out_relation, temp_out_name), compile_plan(plan.right, scidb_out_relation, temp_out_name))
    # print plan.scheme()
    # raise NotImplementedError("Compiling expr of class %s" % plan.__class__)


### HELPER METHODS

def remove_unnamed_literals(plan, expression):
    scheme = plan.scheme()
    ex = str(expression)
    for i in range(len(scheme)):
            unnamed_literal = "$" + str(i)
            repl_str = scheme.getName(i)
            if isinstance(plan, algebra.GroupBy):
                # if isinstance(plan.input, algebra.Apply): # this means I added it there.
                if scheme.getName(i) == '_COLUMN{}_'.format(str(i)):
                    s = str(plan.aggregate_list[i - len(plan.grouping_list)])
                    repl_str = '`{}({})`'.format(s.split('(')[0].lower(), s.split('(')[1][:-1])
                    repl_str = remove_unnamed_literals(plan.input, repl_str) #remove recursive references

            ex = ex.replace(unnamed_literal, repl_str)
    while "$" in ex:
        ex = ex.replace(ex[ex.index('$'):ex.index('$') + 2], plan.scheme().getName(ex.index('$') + 1))
    return ex

def compile_expr(op, child_scheme, state_scheme):
    ####
    # Put special handling at the top!
    ####
    if isinstance(op, expression.NumericLiteral):
        if type(op.value) == int:
            t = types.LONG_TYPE
        elif type(op.value) == float:
            t = types.DOUBLE_TYPE
        else:
            raise NotImplementedError("Compiling NumericLiteral {} of type {}"
                                      .format(op, type(op.value)))

        return str(op.value),

    elif isinstance(op, expression.StringLiteral):
        return str(op.value)
    elif isinstance(op, expression.BooleanLiteral):
        return bool(op.value)
    elif isinstance(op, expression.StateRef):
        return op.get_position(child_scheme, state_scheme)
    elif isinstance(op, expression.AttributeRef):
        return op.get_position(child_scheme, state_scheme)
    elif isinstance(op, expression.Case):
        # Convert n-ary case statements to binary
        op = op.to_binary()
        assert len(op.when_tuples) == 1

        if_expr = compile_expr(op.when_tuples[0][0], child_scheme,
                               state_scheme)
        then_expr = compile_expr(op.when_tuples[0][1], child_scheme,
                                 state_scheme)
        else_expr = compile_expr(op.else_expr, child_scheme, state_scheme)

        return 'iif({},{},{})'.format(if_expr, then_expr, else_expr)
    elif isinstance(op, expression.CAST):
        return "cast({},{})".format(compile_expr(op.input, child_scheme, state_scheme), op._type)

    ####
    # Everything below here is compiled automatically
    ####
    elif isinstance(op, expression.UnaryOperator):
        return "{}({})".format(op.opname(), compile_expr(op.input, child_scheme, state_scheme))
    elif isinstance(op, expression.BinaryOperator):
        return "{} {} {}".format(compile_expr(op.left, child_scheme, state_scheme), op.literals[0], compile_expr(op.right, child_scheme, state_scheme))
    elif isinstance(op, expression.ZeroaryOperator):
        return op.opname()
    elif isinstance(op, expression.NaryOperator):
        children = []
        for operand in op.operands:
            children.append(compile_expr(operand, child_scheme, state_scheme))
        return "{}(){}".format(op.opname(), str(children))
    raise NotImplementedError("Compiling expr of class %s" % op.__class__)
