import logging
import itertools

from raco import rules
from raco.backends import Language, Algebra
from raco import algebra
from raco.expression import *


class SparkLanguage(Language):
    pass


LOGGER = logging.getLogger(__name__)


class SparkOperator(object):
    pass

class SparkScan(algebra.Scan, SparkOperator):
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
        return "filter({}, {})".format(input, remove_unnamed_literals(self.scheme(), self.condition))
        # return "filter({}, {})".format(input, compile_expr(self.condition, self.scheme(), None))

class SparkJoin(algebra.Join, SparkOperator):
    def compileme(self, left, right):
        return "join({},{})".format(left, right)

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
        return "apply({}, {})".format(input, ",".join([remove_unnamed_literals(self.input.scheme(), str(x))
                                                       for x in list(itertools.chain(*self.emitters))]))

### RULES

class GroupByAndJoinToMult(rules.Rule):
    def fire(self, expr):
        # if not isinstance(expr, algebra.Apply):
        #     return expr
        # if not isinstance(expr.input, algebra.GroupBy):
        #     return expr
        # expr = expr.input
        # if not isinstance(expr, algebra.GroupBy):
        #     return expr
        # if not isinstance(expr.input, algebra.Join):
        #     return expr
        # left_schema = expr.input.left.scheme().get_names()
        # right_schema = expr.input.right.scheme().get_names()
        #
        # # Identified a Join --> GroupBy pair.
        # if len(expr.column_list()) != 3: #checking if groupby on 3 columns, 2 dims and 1 aggregate
        #     return expr
        # attrs = 0
        # agts = 0
        # for c in expr.column_list():
        #     if isinstance(c, NamedAttributeRef):
        #         attrs +=1
        #     elif isinstance(c, SUM):
        #         agts +=1
        # if attrs!=2 and agts!=1:
        #     return expr
        #
        # if len(expr.aggregate_list) != 1:
        #     return expr
        # if not isinstance(expr.aggregate_list[0], SUM):
        #     return expr
        # if not isinstance(expr.aggregate_list[0].input, TIMES):
        #     return expr
        # if not isinstance(expr.aggregate_list[0].input.left, NamedAttributeRef):
        #     return expr
        # if not isinstance(expr.aggregate_list[0].input.right, NamedAttributeRef):
        #     return expr
        # val_attr1 = expr.aggregate_list[0].input.left.name
        # val_attr2 = expr.aggregate_list[0].input.right.name
        # if val_attr1 in left_schema:
        #     left_schema.remove(val_attr1)
        # if val_attr2 in left_schema:
        #     left_schema.remove(val_attr2)
        # if val_attr1 in right_schema:
        #     right_schema.remove(val_attr1)
        # if val_attr2 in right_schema:
        #     right_schema.remove(val_attr2)
        #
        # if len(left_schema) != len(right_schema) and len(left_schema) !=2:
        #     return expr
        #
        # for c in expr.column_list():
        #     if isinstance(c, NamedAttributeRef):
        #         if c.name in left_schema:
        #             left_schema.remove(c.name)
        #         if c.name in right_schema:
        #             right_schema.remove(c.name)
        #
        # if len(left_schema) != len(right_schema) and len(left_schema) !=1:
        #     return expr
        #
        # # Checking if join predicate is an equality between attributes
        # # For some reason the join predicate has the form ((1=1) and ($2=$4)), the (1=1) condition is useless, ignoring.
        # if not isinstance(expr.input.condition, AND):
        #     print 'Not AND'
        #     return expr
        # if not isinstance(expr.input.condition.right, EQ):
        #     print expr.input.condition
        #     print 'Not EQ'
        #     return expr
        # print remove_unnamed_literals(expr.input.scheme(), expr.input.condition.right)
        # if not isinstance(expr.input.condition.right.left, UnnamedAttributeRef):
        #     print 'Not named ref left'
        #     return expr
        # if not isinstance(expr.input.condition.right.right, UnnamedAttributeRef):
        #     print 'Not named ref right'
        #     return expr
        # left = remove_unnamed_literals(expr.input.scheme(), expr.input.condition.right.left)
        # right = remove_unnamed_literals(expr.input.scheme(), expr.input.condition.right.right)
        #
        # if left in left_schema:
        #     left_schema.remove(left)
        # if right in left_schema:
        #     left_schema.remove(right)
        # if left in right_schema:
        #     right_schema.remove(left)
        # if right in right_schema:
        #     right_schema.remove(right)
        #
        # if len(left_schema) != len(right_schema) and len(left_schema) !=0:
        #     return expr
        #
        # # TODO: This is still not finished,
        # # we still don't know if the arrays involved have a shape that can be multiplied.
        # # as we have no way of distinguishing between dimensions and attributes.
        # print 'All conditions matched, replacing Join--> Groupby with mult'
        #
        # newop = SciDbMult() # this step hopefully keeps the schema intact.
        # newop.copy(expr)
        # # but we need the info about left and right for the mult, so adding it ourselves, hurray dynamic typing!
        # newop.left = expr.input.left
        # newop.right = expr.input.right
        # newop.isMult = True # TODO: THIS IS BAD. But got no other way to let other rules know about this.
        # return newop
        return expr

    def __str__(self):
        return "GroupBy_plus_Join => SciDbMULT"

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

class GroupByToAggregate(rules.BottomUpRule):
    def fire(self, expr):
        # if isinstance(expr, algebra.GroupBy):
        #     if hasattr(expr, 'isMult'):
        #         return expr
        #     # Todo: Assuming for now that the grouping list consists of only dimensions. Fix Later.
        #     scidbagg = SciDBAggregate(expr.grouping_list, expr.aggregate_list, expr.input)
        #     return scidbagg
        return expr

class JoinToSparkJoin(rules.BottomUpRule):
    def fire(self, expr):
        # if isinstance(expr, algebra.Join):
        #     newop = SciDBJoin()
        #     newop.copy(expr)
        #     type_dict = {'LONG_TYPE': 'int64', 'FLOAT_TYPE': 'double'}
        #     template_1darray = "<{dims_attrs}>{new_dimensions}"
        #
        #     dims_attrs = expr.left.scheme().get_names()
        #     types = expr.left.scheme().get_types()
        #     dims_attrs_string = ','.join('{name}:{t}'.format(name=dims_attrs[i], t=type_dict[types[i]])
        #                                  for i in range(0,len(dims_attrs)))
        #     new_dimensions = '[dim_{r}=1:{total_cells},{total_cells},0]'.format(r=random.randint(1, 10000000),
        #                                                                         total_cells=10000)
        #                                                                         # total_cells=expr.left.num_tuples())
        #     newop.templateleft = template_1darray.format(dims_attrs=dims_attrs_string, new_dimensions=new_dimensions)
        #
        #
        #     dims_attrs = expr.right.scheme().get_names()
        #     types = expr.right.scheme().get_types()
        #     dims_attrs_string = ','.join('{name}:{t}'.format(name=dims_attrs[i], t=type_dict[types[i]])
        #                                  for i in range(0,len(dims_attrs)))
        #     new_dimensions = '[dim_{r}=1:{total_cells},{total_cells},0]'.format(r=random.randint(1, 10000000),
        #                                                                         total_cells=10000)
        #                                                                         # total_cells=expr.right.num_tuples())
        #
        #     newop.templateright = template_1darray.format(dims_attrs=dims_attrs_string, new_dimensions=new_dimensions)
        #
        #     filter = SciDBSelect(expr.condition, newop)
        #     return filter
        return expr

    def __str__(self):
        return "Join => SciDBJoin"

### ALGEBRA

class SparkAlgebra(Algebra):
    """ Spark algebra abstract class"""
    language = SparkLanguage

    operators = [
        SparkScan,
        SparkStore,
        SparkConcat,
        SparkSelect,
        SparkApply,
        SparkJoin
    ]

    def opt_rules(self, **kwargs):
        # replace logical operator with its corresponding SciDB operators
        sparkify = [
            rules.OneToOne(algebra.Store, SparkStore),
            rules.OneToOne(algebra.Scan, SparkScan),
            rules.OneToOne(algebra.UnionAll, SparkConcat),
            rules.OneToOne(algebra.Select, SparkSelect),
            rules.OneToOne(algebra.Apply, SparkApply),
            rules.OneToOne(algebra.Project, SparkProject)
            # rules.OneToOne(algebra.Join, SciDBJoin)
        ]
        all_rules = sparkify + [GroupByAndJoinToMult(), GroupByToAggregate(), JoinToSparkJoin(), ApplyToApplyProject()]

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
