import logging
import itertools

from raco import rules
from raco.backends import Language, Algebra
from raco import algebra
from raco.expression import *


class SciDBLanguage(Language):
    pass


LOGGER = logging.getLogger(__name__)


class SciDBOperator(object):
    pass


class SciDBScan(algebra.Scan, SciDBOperator):
    def compileme(self):
        return "scan({})".format(self.name)


class SciDBStore(algebra.Store, SciDBOperator):
    def compileme(self, inputid):
        return "store({},{})".format(self.relation_key, inputid)


class SciDBConcat(algebra.UnionAll, SciDBOperator):
    def compileme(self, leftid, rightid):
        return "concat({},{})".format(leftid, rightid)


class SciDBSelect(algebra.Select, SciDBOperator):
    def compileme(self, input):
        cond_str = str(self.condition)

        for i in range(len(self.scheme())):
            unnamed_literal = "$" + str(i)
            cond_str = cond_str.replace(unnamed_literal, self.scheme().getName(i))
        return "filter({}, {})".format(input, cond_str)
        # return "filter({}, {})".format(input, compile_expr(self.condition, self.scheme(), None))


class SciDBProject(algebra.Project, SciDBOperator):
    def compileme(self, input):
        return "project({}, {})".format(input, ", ".join([x for x in self.columnlist]))


class SciDBApply(algebra.Apply, SciDBOperator):
    def compileme(self, input):
        return "apply({}, {})".format(input, ",".join([str(x) for x in list(itertools.chain(*self.emitters))]))


class SciDBRedimension(algebra.GroupBy, SciDBOperator):
    @staticmethod
    # TODO: Possible duplication of code.
    def agg_mapping(agg_expr):
        """Maps a BuiltinAggregateExpression to a SciDB string constant
        representing the corresponding aggregate operation."""
        #TODO: Supporting bare mininum expressions for the regrid in our query, needs to be made generic
        if isinstance(agg_expr, raco.expression.BIN):
            return "BIN" + str(int(math.pow(2, agg_expr.n)))
        elif isinstance(agg_expr, raco.expression.SIGNED_COUNT):
            return "signed_count(bucket) as value"
        elif isinstance(agg_expr, raco.expression.AVG):
            return "AVG"
        raise NotImplementedError("SciDBRegrid.agg_mapping({})".format(
            type(agg_expr)))

    def compileme(self, inputid, outputid):
        # TODO: can be an aggregate on attributes or dimensions. Fix later to recognize this distinction
        built_ins = [agg_expr for agg_expr in self.aggregate_list
                     if isinstance(agg_expr, raco.expression.BuiltinAggregateExpression)]

        aggregators = []
        for i, agg_expr in enumerate(built_ins):
            aggregators.append("{}".format(SciDBRegrid.agg_mapping(agg_expr)))

        return "redimension(store({},{}),{},{})".format(inputid, outputid, self.template_array, ",".join(aggregators))

    def shortStr(self):
        return super(SciDBRedimension, self).shortStr() + 'Parent Apply:' + self.parent_apply.shortStr()


class SciDBRegrid(algebra.GroupBy, SciDBOperator):
    @staticmethod
    def agg_mapping(agg_expr):
        """Maps a BuiltinAggregateExpression to a SciDB string constant
        representing the corresponding aggregate operation."""
        # TODO: Supporting bare mininum expressions for the regrid in our query, needs to be made generic
        if isinstance(agg_expr, raco.expression.BIN):
            return "BIN" + str(int(math.pow(2, agg_expr.n)))
        elif isinstance(agg_expr, raco.expression.SIGNED_COUNT):
            return "signed_count(bucket) as value"
        elif isinstance(agg_expr, raco.expression.AVG):
            return "AVG"
        raise NotImplementedError("SciDBRegrid.agg_mapping({})".format(
            type(agg_expr)))


    def compileme(self, inputid):
        group_fields = [ref for ref in self.grouping_list]

        built_ins = [agg_expr for agg_expr in self.aggregate_list
                     if isinstance(agg_expr, raco.expression.BuiltinAggregateExpression)]

        aggregators = []
        for i, agg_expr in enumerate(built_ins):
            aggregators.append("{}({})".format(SciDBRegrid.agg_mapping(agg_expr), 'value'))

        # TODO: What about UDAs? Build support later on. Or since we are converting plans to scidb, is it necessary?
        return "regrid({},{},{})".format(inputid, ",".join(group_fields), ",".join(aggregators))

    def shortStr(self):
        return super(SciDBRegrid, self).shortStr() + 'Parent Apply:' + self.parent_apply.shortStr()


class SciDBAFLAlgebra(Algebra):
    """ SciDB algebra abstract class"""
    language = SciDBLanguage

    operators = [
        SciDBScan,
        SciDBStore,
        SciDBConcat,
        SciDBRegrid,
        SciDBSelect,
        # SciDBProject,
        SciDBApply,
        SciDBRedimension
    ]
    """SciDB physical algebra"""

    def opt_rules(self, **kwargs):
        # replace logical operator with its corresponding SciDB operators
        scidbify = [
            rules.OneToOne(algebra.Store, SciDBStore),
            rules.OneToOne(algebra.Scan, SciDBScan),
            rules.OneToOne(algebra.UnionAll, SciDBConcat),
            rules.OneToOne(algebra.Select, SciDBSelect),
            rules.OneToOne(algebra.Apply, SciDBApply),
            rules.OneToOne(algebra.Project, SciDBProject)
        ]
        all_rules = scidbify + [GroupByToRegridOrRedminension(), ApplyToApplyProject()]

        return all_rules

    def __init__(self, catalog=None):
        self.catalog = catalog


'''
class CountToDimensions(rules.Rule):
    pass
    # select max(i) from X => dimensions(X)[0]
'''
'''
class GroupByToRegrid(rules.Rule):
    def fire(self, expr):
        # if GroupBy
        # Look for dimension calculation:
        #   floor(dim1 / size1)
        #   (assumes that dim is 0:N)
        #   (assumes that dim is 0:N)
'''


class GroupByToRegridOrRedminension(rules.Rule):
    def fire(self, expr):
        # Look for a GroupBy-Apply pair.
        if isinstance(expr, algebra.Apply):
            # Check if the input to the operator is a groupby
            childop = expr.input
            if isinstance(childop, algebra.GroupBy):
                # Looking for dim / size operation.
                # Todo: Crazy amount of hardcoding. Fix Later

                is_regrid = False
                for gb_operation in childop.grouping_list:
                    if isinstance(gb_operation, CAST):
                        if isinstance(gb_operation.input, FLOOR):
                            if isinstance(gb_operation.input.input, DIVIDE):
                                if isinstance(gb_operation.input.input.right, NumericLiteral):
                                    is_regrid = True

                if is_regrid:
                    scidb_regridop = SciDBRegrid(childop.grouping_list, childop.aggregate_list, childop.input)
                    scidb_regridop.parent_apply = algebra.Apply()
                    scidb_regridop.parent_apply.copy(expr)
                    bin_n = 0
                    bin = BIN('value')
                    bin.n = bin_n
                    scidb_regridop.aggregate_list = [AVG('value'), bin]
                    scidb_regridop.grouping_list = ["1", "2"]
                    return scidb_regridop
                else:
                    scidb_redimension = SciDBRedimension(childop.grouping_list, childop.aggregate_list, childop.input)
                    scidb_redimension.aggregate_list = [SIGNED_COUNT(('bucket'))]
                    scidb_redimension.parent_apply = algebra.Apply()
                    scidb_redimension.parent_apply.copy(expr)
                    return scidb_redimension
        return expr

    def __str__(self):
        return "GroupBy => ReGrid/ReDimension"


class ApplyToApplyProject(rules.BottomUpRule):
    def fire(self, expr):
        if isinstance(expr, SciDBApply):
            # Checking for just projection operation.
            just_project = True
            for (n, ex) in expr.emitters:
                if isinstance(ex, NamedAttributeRef):
                    if n != ex.name:
                        just_project = False
            return SciDBProject([name for (name, expr_to_apply) in expr.emitters], expr.input) if just_project else SciDBProject([name for (name, expr_to_apply) in expr.emitters], expr)
        return expr

    def __str__(self):
        return "SciDBApply => SciDBApply followed by a SciDbProject"


def compile_to_afl(plan):
    # TODO Harcoded plan we wan't later we would want the actual conversion.

    ret = """
create temp array transform_1_{r}<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:127,256,0];
create temp array transform_2_{r}<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:63,256,0];
create temp array transform_3_{r}<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:31,256,0];
create temp array transform_4_{r}<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:15,256,0];
create temp array transform_5_{r}<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:7,256,0];
create temp array transform_6_{r}<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:3,256,0];
create temp array transform_7_{r}<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:1,256,0];
create temp array transform_8_{r}<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:0,256,0];

create temp array out_transform_1_{r}<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_2_{r}<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_3_{r}<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_4_{r}<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_5_{r}<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_6_{r}<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_7_{r}<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_8_{r}<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];

save(
  redimension(
    store(
      regrid(
        scan(SciDB__Demo__Vectors),
        1, 2,
        avg(value), bin1(value)),
        transform_1_{r}),
    out_transform_1_{r},
    signed_count(bucket) as value),
  'out/transform_1', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_1_{r}),
        1, 2,
        avg(value), bin2(value)),
        transform_2_{r}),
    out_transform_2_{r},
    signed_count(bucket) as value),
  'out/transform_2', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_2_{r}),
        1, 2,
        avg(value), bin4(value)),
        transform_3_{r}),
    out_transform_3_{r},
    signed_count(bucket) as value),
  'out/transform_3', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_3_{r}),
        1, 2,
        avg(value), bin8(value)),
        transform_4_{r}),
    out_transform_4_{r},
    signed_count(bucket) as value),
  'out/transform_4', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_4_{r}),
        1, 2,
        avg(value), bin16(value)),
        transform_5_{r}),
    out_transform_5_{r},
    signed_count(bucket) as value),
  'out/transform_5', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_5_{r}),
        1, 2,
        avg(value), bin32(value)),
        transform_6_{r}),
    out_transform_6_{r},
    signed_count(bucket) as value),
  'out/transform_6', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_6_{r}),
        1, 2,
        avg(value), bin64(value)),
        transform_7_{r}),
    out_transform_7_{r},
    signed_count(bucket) as value),
  'out/transform_7', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_7_{r}),
        1, 2,
        avg(value), bin128(value)),
        transform_8_{r}),
    out_transform_8_{r},
    signed_count(bucket) as value),
  'out/transform_8', -1, 'csv+');
	""".format(r=random.randint(1, 10000000))
    return ret


def compile_to_afl_new(plan):
    queue = [plan]
    while len(queue) > 0:
        curr_op = queue.pop()
        if isinstance(curr_op, SciDBScan):
            input_relation = str(curr_op.relation_key.relation)
        if isinstance(curr_op, SciDBStore):
            scidb_out_relation = str(curr_op.relation_key).replace(':', '__') # SciDB has a problem with ':'
        for c in curr_op.children():
            queue.append(c)
    temp_out_name = 'out_' + scidb_out_relation

    ret = compile_plan(plan, input_relation, scidb_out_relation, temp_out_name)
    print ret


def compile_plan(plan, input_relation, scidb_out_relation, temp_out_name):
    if isinstance(plan, SciDBStore):
        return "\nsave(" + compile_plan(plan.input, input_relation, scidb_out_relation, temp_out_name) + \
              ", '{scidb_out_relation}', -1, 'csv+');".format(scidb_out_relation=scidb_out_relation)
    if isinstance(plan, SciDBRegrid):
        return plan.compileme(compile_plan(plan.input, input_relation, scidb_out_relation, temp_out_name))
    if isinstance(plan, SciDBRedimension):
        plan.template_array = temp_out_name
        return plan.compileme(compile_plan(plan.input, input_relation, scidb_out_relation, temp_out_name),
                             scidb_out_relation)
    if isinstance(plan, (SciDBSelect, SciDBProject, SciDBApply)):
        return plan.compileme(compile_plan(plan.input, input_relation, scidb_out_relation, temp_out_name))
    if isinstance(plan, SciDBScan):
        return "scan({input_relation})".format(input_relation=input_relation)

    raise NotImplementedError()

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