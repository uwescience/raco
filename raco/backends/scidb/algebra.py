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
        if hasattr(self, 'flatScan'):
            if self.flatScan:
                return "redimension({},{})".format(self.name, self.template)
        else:
            return "scan({})".format(str(self.relation_key).split(':')[-1])

class SciDbMult(algebra.GroupBy, SciDBOperator):
    def compileme(self, leftsym, rightsym):
        return "multiply({},{})".format(leftsym, rightsym)

class SciDBStore(algebra.Store, SciDBOperator):
    def compileme(self, inputid):
        return "store({},{})".format(self.relation_key, inputid)

class SciDBConcat(algebra.UnionAll, SciDBOperator):
    def compileme(self, leftid, rightid):
        return "concat({},{})".format(leftid, rightid)

class SciDBSelect(algebra.Select, SciDBOperator):
    def compileme(self, input):
        return "filter({}, {})".format(input, remove_unnamed_literals(self.scheme(), self.condition))
        # return "filter({}, {})".format(input, compile_expr(self.condition, self.scheme(), None))

class SciDBJoin(algebra.Join, SciDBOperator):
    def compileme(self, left, right):
        return "join(redimension({},{}),redimension({},{}))".format(left, self.templateleft, right, self.templateright)

class SciDBProject(algebra.Project, SciDBOperator):
    def compileme(self, input):
        return "project(redimension({},{}), {})".format(input, self.redimtemplate, ", ".join([x.name for x in self.columnlist]))

class SciDBAggregate(algebra.GroupBy, SciDBOperator):
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

class SciDBApply(algebra.Apply, SciDBOperator):
    def compileme(self, input):
        return "apply({}, {})".format(input, ",".join([remove_unnamed_literals(self.input.scheme(), str(x))
                                                       for x in list(itertools.chain(*self.emitters))]))

class SciDBRedimension(algebra.UnaryOperator, SciDBOperator):
    def compileme(self, input):
        return "redimension({},{})".format(input, self.template)

# class SciDBRedimension(algebra.GroupBy, SciDBOperator):
#     @staticmethod
#     # TODO: Possible duplication of code.
#     def agg_mapping(agg_expr):
#         """Maps a BuiltinAggregateExpression to a SciDB string constant
#         representing the corresponding aggregate operation."""
#         #TODO: Supporting bare mininum expressions for the regrid in our query, needs to be made generic
#         if isinstance(agg_expr, raco.expression.BIN):
#             return "BIN" + str(int(math.pow(2, agg_expr.n)))
#         elif isinstance(agg_expr, raco.expression.SIGNED_COUNT):
#             return "signed_count(bucket) as value"
#         elif isinstance(agg_expr, raco.expression.AVG):
#             return "AVG"
#         raise NotImplementedError("SciDBRegrid.agg_mapping({})".format(
#             type(agg_expr)))
#
#     def compileme(self, inputid, outputid):
#         # TODO: can be an aggregate on attributes or dimensions. Fix later to recognize this distinction
#         built_ins = [agg_expr for agg_expr in self.aggregate_list
#                      if isinstance(agg_expr, raco.expression.BuiltinAggregateExpression)]
#
#         aggregators = []
#         for i, agg_expr in enumerate(built_ins):
#             aggregators.append("{}".format(SciDBRegrid.agg_mapping(agg_expr)))
#
#         return "redimension(store({},{}),{},{})".format(inputid, outputid, self.template_array, ",".join(aggregators))
#
#     def shortStr(self):
#         return super(SciDBRedimension, self).shortStr() + 'Parent Apply:' + self.parent_apply.shortStr()


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

### RULES

class ScanToFlatScan(rules.Rule):
    def fire(self, expr):
        if isinstance(expr, algebra.Scan):
            newop = SciDBScan()
            newop.copy(expr)
            newop.flatScan = True
            type_dict = {'LONG_TYPE': 'int64', 'FLOAT_TYPE': 'double'}
            template_1darray = "<{dims_attrs}>{new_dimensions}"

            dims_attrs = newop.scheme().get_names()
            types = newop.scheme().get_types()
            dims_attrs_string = ','.join('{name}:{t}'.format(name=dims_attrs[i], t=type_dict[types[i]])
                                         for i in range(0,len(dims_attrs)))
            new_dimensions = '[dim_{r}=1:{total_cells},{total_cells},0]'.format(r=random.randint(1, 10000000),
                                                                                total_cells=newop._cardinality[1])
            newop.template = template_1darray.format(dims_attrs=dims_attrs_string, new_dimensions=new_dimensions)
            newop.name = str(newop.relation_key).split(':')[-1]
            return newop
        return expr

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

class GroupByAndJoinToMult(rules.Rule):
    def fire(self, expr):
        if not isinstance(expr, algebra.GroupBy):
            return expr
        if not isinstance(expr.input, algebra.Join):
            return expr
        left_schema = expr.input.left.scheme().get_names()
        right_schema = expr.input.right.scheme().get_names()

        # Identified a Join --> GroupBy pair.
        if len(expr.column_list()) != 3: #checking if groupby on 3 columns, 2 dims and 1 aggregate
            return expr
        attrs = 0
        agts = 0
        for c in expr.column_list():
            if isinstance(c, NamedAttributeRef):
                attrs +=1
            elif isinstance(c, SUM):
                agts +=1
        if attrs!=2 and agts!=1:
            return expr

        if len(expr.aggregate_list) != 1:
            return expr
        if not isinstance(expr.aggregate_list[0], SUM):
            return expr
        if not isinstance(expr.aggregate_list[0].input, TIMES):
            return expr
        if not isinstance(expr.aggregate_list[0].input.left, NamedAttributeRef):
            return expr
        if not isinstance(expr.aggregate_list[0].input.right, NamedAttributeRef):
            return expr
        val_attr1 = expr.aggregate_list[0].input.left.name
        val_attr2 = expr.aggregate_list[0].input.right.name
        if val_attr1 in left_schema:
            left_schema.remove(val_attr1)
        if val_attr2 in left_schema:
            left_schema.remove(val_attr2)
        if val_attr1 in right_schema:
            right_schema.remove(val_attr1)
        if val_attr2 in right_schema:
            right_schema.remove(val_attr2)

        if len(left_schema) != len(right_schema) and len(left_schema) !=2:
            return expr

        for c in expr.column_list():
            if isinstance(c, NamedAttributeRef):
                if c.name in left_schema:
                    left_schema.remove(c.name)
                if c.name in right_schema:
                    right_schema.remove(c.name)

        if len(left_schema) != len(right_schema) and len(left_schema) !=1:
            return expr

        # Checking if join predicate is an equality between attributes
        # For some reason the join predicate has the form ((1=1) and ($2=$4)), the (1=1) condition is useless, ignoring.
        if not isinstance(expr.input.condition, AND):
            print 'Not AND'
            return expr
        if not isinstance(expr.input.condition.right, EQ):
            print expr.input.condition
            print 'Not EQ'
            return expr
        print remove_unnamed_literals(expr.input.scheme(), expr.input.condition.right)
        if not isinstance(expr.input.condition.right.left, UnnamedAttributeRef):
            print 'Not named ref left'
            return expr
        if not isinstance(expr.input.condition.right.right, UnnamedAttributeRef):
            print 'Not named ref right'
            return expr
        left = remove_unnamed_literals(expr.input.scheme(), expr.input.condition.right.left)
        right = remove_unnamed_literals(expr.input.scheme(), expr.input.condition.right.right)

        if left in left_schema:
            left_schema.remove(left)
        if right in left_schema:
            left_schema.remove(right)
        if left in right_schema:
            right_schema.remove(left)
        if right in right_schema:
            right_schema.remove(right)

        if len(left_schema) != len(right_schema) and len(left_schema) !=0:
            return expr

        # TODO: This is still not finished,
        # we still don't know if the arrays involved have a shape that can be multiplied.
        # as we have no way of distinguishing between dimensions and attributes.
        print 'All conditions matched, replacing Join--> Groupby with mult'

        newop = SciDbMult() # this step hopefully keeps the schema intact.
        newop.copy(expr)
        # but we need the info about left and right for the mult, so adding it ourselves, hurray dynamic typing!
        newop.left = expr.input.left
        newop.right = expr.input.right
        newop.isMult = True # TODO: THIS IS BAD. But got no other way to let other rules know about this.
        return newop

    def __str__(self):
        return "GroupBy_plus_Join => SciDbMULT"

class ApplyToApplyProject(rules.BottomUpRule):
    def fire(self, expr):
        if isinstance(expr, SciDBApply):
            # Checking for just projection operation.
            just_project = True
            for (n, ex) in expr.emitters:
                if isinstance(ex, NamedAttributeRef):
                    if n != ex.name:
                        just_project = False
                else:
                    just_project = False
            type_dict = {'LONG_TYPE': 'int64', 'FLOAT_TYPE': 'double'}
            template_1darray = "<{dims_attrs}>{new_dimensions}"

            input_to_project = expr.input if just_project else expr

            dims_attrs = input_to_project.scheme().get_names()
            types = input_to_project.scheme().get_types()
            dims_attrs_string = ','.join('{name}:{t}'.format(name=dims_attrs[i], t=type_dict[types[i]])
                                         for i in range(0,len(dims_attrs)))
            new_dimensions = '[dim_{r}=1:{total_cells},{total_cells},0]'.format(r=random.randint(1, 10000000),
                                                                                total_cells=10000)
                                                                                # total_cells=input_to_project.num_tuples()[1])
            # redim = SciDBRedimension(input_to_project)
            toreturn = SciDBProject([NamedAttributeRef(name) for (name, expr_to_apply) in expr.emitters], input_to_project)
            toreturn.redimtemplate = template_1darray.format(dims_attrs=dims_attrs_string, new_dimensions=new_dimensions)
            return toreturn
        return expr

    def __str__(self):
        return "SciDBApply => SciDBApply followed by a SciDbProject"

class GroupByToAggregate(rules.BottomUpRule):
    def fire(self, expr):
        if isinstance(expr, algebra.GroupBy):
            if hasattr(expr, 'isMult'):
                return expr
            # Todo: Assuming for now that the grouping list consists of only dimensions. Fix Later.
            scidbagg = SciDBAggregate(expr.grouping_list, expr.aggregate_list, expr.input)
            return scidbagg
        return expr

class JoinToSciDBJoin(rules.Rule):
    def fire(self, expr):
        if isinstance(expr, algebra.Join):
            newop = SciDBJoin()
            newop.copy(expr)
            type_dict = {'LONG_TYPE': 'int64', 'FLOAT_TYPE': 'double'}
            template_1darray = "<{dims_attrs}>{new_dimensions}"

            dims_attrs = expr.left.scheme().get_names()
            types = expr.left.scheme().get_types()
            dims_attrs_string = ','.join('{name}:{t}'.format(name=dims_attrs[i], t=type_dict[types[i]])
                                         for i in range(0,len(dims_attrs)))
            new_dimensions = '[dim_{r}=1:{total_cells},{total_cells},0]'.format(r=random.randint(1, 10000000),
                                                                                # total_cells=10000)
                                                                                total_cells=expr.left.num_tuples())
            newop.templateleft = template_1darray.format(dims_attrs=dims_attrs_string, new_dimensions=new_dimensions)


            dims_attrs = expr.right.scheme().get_names()
            types = expr.right.scheme().get_types()
            dims_attrs_string = ','.join('{name}:{t}'.format(name=dims_attrs[i], t=type_dict[types[i]])
                                         for i in range(0,len(dims_attrs)))
            new_dimensions = '[dim_{r}=1:{total_cells},{total_cells},0]'.format(r=random.randint(1, 10000000),
                                                                                # total_cells=10000)
                                                                                total_cells=expr.right.num_tuples())

            newop.templateright = template_1darray.format(dims_attrs=dims_attrs_string, new_dimensions=new_dimensions)
            return newop
        return expr

    def __str__(self):
        return "Join => SciDBJoin"

### ALGEBRA

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
        SciDBRedimension,
        SciDBJoin
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
            # rules.OneToOne(algebra.Join, SciDBJoin)
        ]
        # all_rules = scidbify + [GroupByToRegridOrRedminension(), ApplyToApplyProject()] # Removing the hardcoding rule
        all_rules = scidbify + [GroupByAndJoinToMult(), GroupByToAggregate(), JoinToSciDBJoin(), ApplyToApplyProject()]

        return all_rules

    def __init__(self, catalog=None):
        self.catalog = catalog


### Compilation to CODE
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

    ret = compile_plan(plan, scidb_out_relation, temp_out_name)
    print ret

def compile_plan(plan, scidb_out_relation, temp_out_name):
    if isinstance(plan, SciDBStore):
        return "\nstore(" + compile_plan(plan.input, scidb_out_relation, temp_out_name) + \
              ", {scidb_out_relation});".format(scidb_out_relation=str(plan.relation_key).replace(':', '__'))
    if isinstance(plan, SciDBRegrid):
        return plan.compileme(compile_plan(plan.input, scidb_out_relation, temp_out_name))
    if isinstance(plan, SciDBRedimension):
        # plan.template_array = temp_out_name
        return plan.compileme(compile_plan(plan.input, scidb_out_relation, temp_out_name))
    if isinstance(plan, (SciDBSelect, SciDBProject, SciDBApply)):
        return plan.compileme(compile_plan(plan.input, scidb_out_relation, temp_out_name))
    if isinstance(plan, SciDBScan):
        return plan.compileme()
    if isinstance(plan, SciDBJoin):
        return plan.compileme(compile_plan(plan.left, scidb_out_relation, temp_out_name),
                              compile_plan(plan.right, scidb_out_relation, temp_out_name))
    if isinstance(plan, SciDBAggregate):
        return plan.compileme(compile_plan(plan.input, scidb_out_relation, temp_out_name))
    if isinstance(plan, SciDbMult):
        return plan.compileme(compile_plan(plan.left, scidb_out_relation, temp_out_name), compile_plan(plan.right, scidb_out_relation, temp_out_name))
    print plan.scheme()
    raise NotImplementedError("Compiling expr of class %s" % plan.__class__)


### HELPER METHODS

def remove_unnamed_literals(scheme, expression):
    ex = str(expression)
    for i in range(len(scheme)):
            unnamed_literal = "$" + str(i)
            ex = ex.replace(unnamed_literal, scheme.getName(i))
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
