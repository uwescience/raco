from collections import defaultdict

from raco import algebra
from raco import rules
from raco.scheme import Scheme
from raco import expression
from raco.language import Language
from raco.utility import emit
from raco.relation_key import RelationKey
from raco.expression.aggregate import DecomposableAggregate


def scheme_to_schema(s):
    def convert_typestr(t):
        if t.lower() in ['bool', 'boolean']:
            return 'BOOLEAN_TYPE'
        if t.lower() in ['float', 'double']:
            return 'DOUBLE_TYPE'
#    if t.lower() in ['float']:
#      return 'FLOAT_TYPE'
#    if t.lower() in ['int', 'integer']:
#      return 'INT_TYPE'
        if t.lower() in ['int', 'integer', 'long']:
            return 'LONG_TYPE'
        if t.lower() in ['str', 'string']:
            return 'STRING_TYPE'
        return t
    if s:
        names, descrs = zip(*s.asdict.items())
        names = ["%s" % n for n in names]
        types = [convert_typestr(r[1]) for r in descrs]
    else:
        names = []
        types = []
    return {"columnTypes": types, "columnNames": names}


def compile_expr(op, child_scheme, state_scheme):
    ####
    # Put special handling at the top!
    ####
    if isinstance(op, expression.NumericLiteral):
        if type(op.value) == int:
            if op.value <= (2 ** 31) - 1 and op.value >= -2 ** 31:
                myria_type = 'INT_TYPE'
            else:
                myria_type = 'LONG_TYPE'
        elif type(op.value) == float:
            myria_type = 'DOUBLE_TYPE'
        else:
            raise NotImplementedError("Compiling NumericLiteral %s of type %s" % (op, type(op.value)))  # noqa

        return {
            'type': 'CONSTANT',
            'value': str(op.value),
            'valueType': myria_type
        }
    elif isinstance(op, expression.StringLiteral):
        return {
            'type': 'CONSTANT',
            'value': str(op.value),
            'valueType': 'STRING_TYPE'
        }
    elif isinstance(op, expression.StateRef):
        return {
            'type': 'STATE',
            'columnIdx': op.get_position(child_scheme, state_scheme)
        }
    elif isinstance(op, expression.AttributeRef):
        return {
            'type': 'VARIABLE',
            'columnIdx': op.get_position(child_scheme, state_scheme)
        }
    elif isinstance(op, expression.Case):
        # Convert n-ary case statements to binary, as expected by Myria
        op = op.to_binary()
        assert len(op.when_tuples) == 1

        if_expr = compile_expr(op.when_tuples[0][0], child_scheme,
                               state_scheme)
        then_expr = compile_expr(op.when_tuples[0][1], child_scheme,
                                 state_scheme)
        else_expr = compile_expr(op.else_expr, child_scheme, state_scheme)

        return {
            'type': 'CONDITION',
            'children': [if_expr, then_expr, else_expr]
        }

    ####
    # Everything below here is compiled automatically
    ####
    elif isinstance(op, expression.UnaryOperator):
        return {
            'type': op.opname(),
            'operand': compile_expr(op.input, child_scheme, state_scheme)
        }
    elif isinstance(op, expression.BinaryOperator):
        return {
            'type': op.opname(),
            'left': compile_expr(op.left, child_scheme, state_scheme),
            'right': compile_expr(op.right, child_scheme, state_scheme)
        }

    raise NotImplementedError("Compiling expr of class %s" % op.__class__)


def compile_mapping(expr, child_scheme, state_scheme):
    output_name, root_op = expr
    return {
        'outputName': output_name,
        'rootExpressionOperator': compile_expr(root_op,
                                               child_scheme,
                                               state_scheme)
    }


class MyriaLanguage(Language):
    reusescans = False

    @classmethod
    def new_relation_assignment(cls, rvar, val):
        return emit(cls.relation_decl(rvar), cls.assignment(rvar, val))

    @classmethod
    def relation_decl(cls, rvar):
        # no type declarations necessary
        return ""

    @staticmethod
    def assignment(x, y):
        return ""

    @staticmethod
    def comment(txt):
        # comments not technically allowed in json
        return ""

    @classmethod
    def boolean_combine(cls, args, operator="and"):
        opstr = " %s " % operator
        conjunc = opstr.join(["%s" % arg for arg in args])
        return "(%s)" % conjunc

    @staticmethod
    def mklambda(body, var="t"):
        return ("lambda %s: " % var) + body

    @staticmethod
    def compile_attribute(name):
        return '%s' % name


class MyriaOperator(object):
    language = MyriaLanguage


class MyriaScan(algebra.Scan, MyriaOperator):
    def compileme(self, resultsym):
        return {
            "opName": resultsym,
            "opType": "TableScan",
            "relationKey": {
                "userName": self.relation_key.user,
                "programName": self.relation_key.program,
                "relationName": self.relation_key.relation
            }
        }


class MyriaScanTemp(algebra.ScanTemp, MyriaOperator):
    def compileme(self, resultsym):
        return {
            "opName": resultsym,
            "opType": "TableScan",
            "relationKey": {
                "userName": 'public',
                "programName": '__TEMP__',
                "relationName": self.name
            }
        }


class MyriaUnionAll(algebra.UnionAll, MyriaOperator):
    def compileme(self, resultsym, leftsym, rightsym):
        return {
            "opName": resultsym,
            "opType": "UnionAll",
            "argChildren": [leftsym, rightsym]
        }


class MyriaSingleton(algebra.SingletonRelation, MyriaOperator):
    def compileme(self, resultsym):
        return {
            "opName": resultsym,
            "opType": "Singleton",
        }


class MyriaEmptyRelation(algebra.EmptyRelation, MyriaOperator):
    def compileme(self, resultsym):
        return {
            "opName": resultsym,
            "opType": "Empty",
            'schema': scheme_to_schema(self.scheme())
        }


class MyriaSelect(algebra.Select, MyriaOperator):
    def compileme(self, resultsym, inputsym):
        pred = compile_expr(self.condition, self.scheme(), None)
        return {
            "opName": resultsym,
            "opType": "Filter",
            "argChild": inputsym,
            "argPredicate": {
                "rootExpressionOperator": pred
            }
        }


class MyriaCrossProduct(algebra.CrossProduct, MyriaOperator):
    def compileme(self, resultsym, leftsym, rightsym):
        column_names = [name for (name, _) in self.scheme()]
        allleft = [i.position for i in self.left.scheme().ascolumnlist()]
        allright = [i.position for i in self.right.scheme().ascolumnlist()]
        return {
            "opName": resultsym,
            "opType": "SymmetricHashJoin",
            "argColumnNames": column_names,
            "argChild1": leftsym,
            "argChild2": rightsym,
            "argColumns1": [],
            "argColumns2": [],
            "argSelect1": allleft,
            "argSelect2": allright
        }


class MyriaStore(algebra.Store, MyriaOperator):
    def compileme(self, resultsym, inputsym):
        return {
            "opName": resultsym,
            "opType": "DbInsert",
            "relationKey": {
                "userName": self.relation_key.user,
                "programName": self.relation_key.program,
                "relationName": self.relation_key.relation
            },
            "argOverwriteTable": True,
            "argChild": inputsym,
        }


class MyriaStoreTemp(algebra.StoreTemp, MyriaOperator):
    def compileme(self, resultsym, inputsym):
        return {
            "opName": resultsym,
            "opType": "DbInsert",
            "relationKey": {
                "userName": 'public',
                "programName": '__TEMP__',
                "relationName": self.name
            },
            "argOverwriteTable": True,
            "argChild": inputsym,
        }


def convertcondition(condition, left_len, combined_scheme):
    """Convert an equijoin condition to a pair of column lists."""

    if isinstance(condition, expression.AND):
        leftcols1, rightcols1 = convertcondition(condition.left,
                                                 left_len,
                                                 combined_scheme)
        leftcols2, rightcols2 = convertcondition(condition.right,
                                                 left_len,
                                                 combined_scheme)
        return leftcols1 + leftcols2, rightcols1 + rightcols2

    if isinstance(condition, expression.EQ):
        leftpos = condition.left.get_position(combined_scheme)
        rightpos = condition.right.get_position(combined_scheme)
        leftcol = min(leftpos, rightpos)
        rightcol = max(leftpos, rightpos)
        assert rightcol >= left_len
        return [leftcol], [rightcol - left_len]

    raise NotImplementedError("Myria only supports EquiJoins, not %s" % condition)  # noqa


class MyriaSymmetricHashJoin(algebra.ProjectingJoin, MyriaOperator):

    def compileme(self, resultsym, leftsym, rightsym):
        """Compile the operator to a sequence of json operators"""

        left_len = len(self.left.scheme())
        combined = self.left.scheme() + self.right.scheme()
        leftcols, rightcols = convertcondition(self.condition,
                                               left_len,
                                               combined)

        if self.columnlist is None:
            self.columnlist = self.scheme().ascolumnlist()
        column_names = [name for (name, _) in self.scheme()]

        pos = [i.get_position(combined) for i in self.columnlist]
        allleft = [i for i in pos if i < left_len]
        allright = [i - left_len for i in pos if i >= left_len]

        join = {
            "opName": resultsym,
            "opType": "SymmetricHashJoin",
            "argColumnNames": column_names,
            "argChild1": "%s" % leftsym,
            "argColumns1": leftcols,
            "argChild2": "%s" % rightsym,
            "argColumns2": rightcols,
            "argSelect1": allleft,
            "argSelect2": allright
        }

        return join


class MyriaGroupBy(algebra.GroupBy, MyriaOperator):
    @staticmethod
    def agg_mapping(agg_expr):
        """Maps an AggregateExpression to a Myria string constant representing
        the corresponding aggregate operation."""
        if isinstance(agg_expr, expression.MAX):
            return "AGG_OP_MAX"
        elif isinstance(agg_expr, expression.MIN):
            return "AGG_OP_MIN"
        elif isinstance(agg_expr, expression.COUNT):
            return "AGG_OP_COUNT"
        elif isinstance(agg_expr, expression.COUNTALL):
            return "AGG_OP_COUNT"  # XXX Wrong in the presence of nulls
        elif isinstance(agg_expr, expression.SUM):
            return "AGG_OP_SUM"

    def compileme(self, resultsym, inputsym):
        child_scheme = self.input.scheme()
        group_fields = [expression.toUnnamed(ref, child_scheme)
                        for ref in self.grouping_list]

        agg_fields = []
        for expr in self.aggregate_list:
            if isinstance(expr, expression.COUNTALL):
                # XXX Wrong in the presence of nulls
                agg_fields.append(expression.UnnamedAttributeRef(0))
            else:
                agg_fields.append(expression.toUnnamed(expr.input,
                    child_scheme))

        agg_types = [[MyriaGroupBy.agg_mapping(agg_expr)]
                     for agg_expr in self.aggregate_list]
        ret = {
            "opName": resultsym,
            "argChild": inputsym,
            "argAggFields": [agg_field.position for agg_field in agg_fields],
            "argAggOperators": agg_types,
        }

        num_fields = len(self.grouping_list)
        if num_fields == 0:
            ret["opType"] = "Aggregate"
        elif num_fields == 1:
            ret["opType"] = "SingleGroupByAggregate"
            ret["argGroupField"] = group_fields[0].position
        else:
            ret["opType"] = "MultiGroupByAggregate"
            ret["argGroupFields"] = [field.position for field in group_fields]
        return ret


class MyriaShuffle(algebra.Shuffle, MyriaOperator):
    """Represents a simple shuffle operator"""
    def compileme(self, resultsym, inputsym):
        raise NotImplementedError('shouldn''t ever get here, should be turned into SP-SC pair')  # noqa


class MyriaCollect(algebra.Collect, MyriaOperator):
    """Represents a simple collect operator"""
    def compileme(self, resultsym, inputsym):
        raise NotImplementedError('shouldn''t ever get here, should be turned into CP-CC pair')  # noqa


class MyriaDupElim(algebra.Distinct, MyriaOperator):
    """Represents duplicate elimination"""
    def compileme(self, resultsym, inputsym):
        return {
            "opName": resultsym,
            "opType": "DupElim",
            "argChild": inputsym,
        }


class MyriaApply(algebra.Apply, MyriaOperator):
    """Represents a simple apply operator"""
    def compileme(self, resultsym, inputsym):
        child_scheme = self.input.scheme()
        emitters = [compile_mapping(x, child_scheme, None)
                    for x in self.emitters]
        return {
            'opName': resultsym,
            'opType': 'Apply',
            'argChild': inputsym,
            'emitExpressions': emitters
        }


class MyriaStatefulApply(algebra.StatefulApply, MyriaOperator):
    """Represents a stateful apply operator"""
    def compileme(self, resultsym, inputsym):
        child_scheme = self.input.scheme()
        state_scheme = self.state_scheme
        comp_map = lambda x: compile_mapping(x, child_scheme, state_scheme)
        emitters = [comp_map(x) for x in self.emitters]
        inits = [comp_map(x) for x in self.inits]
        updaters = [comp_map(x) for x in self.updaters]
        return {
            'opName': resultsym,
            'opType': 'StatefulApply',
            'argChild': inputsym,
            'emitExpressions': emitters,
            'initializerExpressions': inits,
            'updaterExpressions': updaters
        }


class MyriaBroadcastProducer(algebra.UnaryOperator, MyriaOperator):
    """A Myria BroadcastProducer"""
    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)

    def shortStr(self):
        return "%s" % self.opname()

    def compileme(self, resultsym, inputsym):
        return {
            "opName": resultsym,
            "opType": "BroadcastProducer",
            "argChild": inputsym,
        }


class MyriaBroadcastConsumer(algebra.UnaryOperator, MyriaOperator):
    """A Myria BroadcastConsumer"""
    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)

    def shortStr(self):
        return "%s" % self.opname()

    def compileme(self, resultsym, inputsym):
        return {
            'opName': resultsym,
            'opType': 'BroadcastConsumer',
            'argOperatorId': inputsym
        }


class MyriaShuffleProducer(algebra.UnaryOperator, MyriaOperator):
    """A Myria ShuffleProducer"""
    def __init__(self, input, hash_columns):
        algebra.UnaryOperator.__init__(self, input)
        self.hash_columns = hash_columns

    def shortStr(self):
        hash_string = ','.join([str(x) for x in self.hash_columns])
        return "%s(h(%s))" % (self.opname(), hash_string)

    def compileme(self, resultsym, inputsym):
        if len(self.hash_columns) == 1:
            pf = {
                "type": "SingleFieldHash",
                "index": self.hash_columns[0]
            }
        else:
            pf = {
                "type": "MultiFieldHash",
                "indexes": self.hash_columns
            }

        return {
            "opName": resultsym,
            "opType": "ShuffleProducer",
            "argChild": inputsym,
            "argPf": pf
        }


class MyriaShuffleConsumer(algebra.UnaryOperator, MyriaOperator):
    """A Myria ShuffleConsumer"""
    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)

    def shortStr(self):
        return "%s" % self.opname()

    def compileme(self, resultsym, inputsym):
        return {
            'opName': resultsym,
            'opType': 'ShuffleConsumer',
            'argOperatorId': inputsym
        }


class BreakShuffle(rules.Rule):
    def fire(self, expr):
        if not isinstance(expr, MyriaShuffle):
            return expr

        producer = MyriaShuffleProducer(expr.input, expr.columnlist)
        consumer = MyriaShuffleConsumer(producer)
        return consumer


class MyriaCollectProducer(algebra.UnaryOperator, MyriaOperator):
    """A Myria CollectProducer"""
    def __init__(self, input, server):
        algebra.UnaryOperator.__init__(self, input)
        self.server = server

    def shortStr(self):
        return "%s(@%s)" % (self.opname(), self.server)

    def compileme(self, resultsym, inputsym):
        return {
            "opName": resultsym,
            "opType": "CollectProducer",
            "argChild": inputsym,
        }


class MyriaCollectConsumer(algebra.UnaryOperator, MyriaOperator):
    """A Myria CollectConsumer"""
    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)

    def shortStr(self):
        return "%s" % self.opname()

    def compileme(self, resultsym, inputsym):
        return {
            'opName': resultsym,
            'opType': 'CollectConsumer',
            'argOperatorId': inputsym
        }


class BreakCollect(rules.Rule):
    def fire(self, expr):
        if not isinstance(expr, MyriaCollect):
            return expr

        producer = MyriaCollectProducer(expr.input, None)
        consumer = MyriaCollectConsumer(producer)
        return consumer


class BreakBroadcast(rules.Rule):
    def fire(self, expr):
        if not isinstance(expr, algebra.Broadcast):
            return expr

        producer = MyriaBroadcastProducer(expr.input)
        consumer = MyriaBroadcastConsumer(producer)
        return consumer


class ShuffleBeforeJoin(rules.Rule):
    def fire(self, expr):
        # If not a join, who cares?
        if not isinstance(expr, algebra.Join):
            return expr

        # If both have shuffles already, who cares?
        if (isinstance(expr.left, algebra.Shuffle)
                and isinstance(expr.right, algebra.Shuffle)):
            return expr

        # Figure out which columns go in the shuffle
        left_cols, right_cols = \
            convertcondition(expr.condition,
                             len(expr.left.scheme()),
                             expr.left.scheme() + expr.right.scheme())

        # Left shuffle
        if isinstance(expr.left, algebra.Shuffle):
            left_shuffle = expr.left
        else:
            left_shuffle = algebra.Shuffle(expr.left, left_cols)
        # Right shuffle
        if isinstance(expr.right, algebra.Shuffle):
            right_shuffle = expr.right
        else:
            right_shuffle = algebra.Shuffle(expr.right, right_cols)

        # Construct the object!
        if isinstance(expr, algebra.ProjectingJoin):
            return algebra.ProjectingJoin(expr.condition,
                                          left_shuffle, right_shuffle,
                                          expr.columnlist)
        elif isinstance(expr, algebra.Join):
            return algebra.Join(expr.condition, left_shuffle, right_shuffle)
        raise NotImplementedError("How the heck did you get here?")


class BroadcastBeforeCross(rules.Rule):
    def fire(self, expr):
        # If not a CrossProduct, who cares?
        if not isinstance(expr, algebra.CrossProduct):
            return expr

        if isinstance(expr.left, algebra.Broadcast) or \
                isinstance(expr.right, algebra.Broadcast):
            return expr

        # By default, broadcast the right child
        expr.right = algebra.Broadcast(expr.right)

        return expr


class DistributedGroupBy(rules.Rule):

    @staticmethod
    def do_transfer(op):
        """Introduce a network transfer before a groupby operation."""

        # Get an array of position references to columns in the child scheme
        child_scheme = op.input.scheme()
        group_fields = [expression.toUnnamed(ref, child_scheme).position
                        for ref in op.grouping_list]
        if len(group_fields) == 0:
            # Need to Collect all tuples at once place
            op.input = algebra.Collect(op.input)
        else:
            # Need to Shuffle
            op.input = algebra.Shuffle(op.input, group_fields)

        return op

    def fire(self, op):
        # If not a GroupBy, who cares?
        if op.__class__ != algebra.GroupBy:
            return op

        num_grouping_terms = len(op.grouping_list)
        decomposable_aggs = [agg for agg in op.aggregate_list if
                             isinstance(agg, DecomposableAggregate)]

        # All built-in aggregates are now decomposable
        assert len(decomposable_aggs) == len(op.aggregate_list)

        # Each logical aggregate generates one or more local aggregates:
        # e.g., average requires a SUM and a COUNT.  In turn, these local
        # aggregates are consumed by merge aggregates.

        local_aggs = []   # aggregates executed on each local machine
        merge_aggs = []   # aggregates executed after local aggs
        agg_offsets = []  # map from logical index to local/merge index.

        for logical_agg in op.aggregate_list:
            agg_offsets.append(len(local_aggs))
            local_aggs.extend(logical_agg.get_local_aggregates())
            merge_aggs.extend(logical_agg.get_merge_aggregates())

        assert len(merge_aggs) == len(local_aggs)

        local_gb = MyriaGroupBy(op.grouping_list, local_aggs, op.input)

        # Create a merge aggregate; grouping terms are passed through.
        merge_groupings = [expression.UnnamedAttributeRef(i)
                           for i in range(num_grouping_terms)]

        # Connect the output of local aggregates to merge aggregates
        for pos, agg in enumerate(merge_aggs, num_grouping_terms):
            agg.input = expression.UnnamedAttributeRef(pos)

        merge_gb = MyriaGroupBy(merge_groupings, merge_aggs, local_gb)
        op_out = self.do_transfer(merge_gb)

        # Extract a single result per logical aggregate using the finalizer
        # expressions (if any)
        has_finalizer = any([agg.get_finalizer() for agg in op.aggregate_list])
        if not has_finalizer:
            return op_out

        def resolve_finalizer_expr(logical_agg, pos):
            assert isinstance(logical_agg, DecomposableAggregate)
            fexpr = logical_agg.get_finalizer()

            # Start of merge aggregates for this logical aggregate
            offset = num_grouping_terms + agg_offsets[pos]

            if fexpr is None:
                return expression.UnnamedAttributeRef(offset)
            else:
                # Convert MergeAggregateOutput instances to absolute col refs
                return expression.finalizer_expr_to_absolute(fexpr, offset)

        # pass through grouping terms
        gmappings = [(None, expression.UnnamedAttributeRef(i))
                     for i in range(len(op.grouping_list))]
        # extract a single result for aggregate terms
        fmappings = [(None, resolve_finalizer_expr(agg, pos)) for pos, agg in
                     enumerate(op.aggregate_list)]
        return algebra.Apply(gmappings + fmappings, op_out)


class SplitSelects(rules.Rule):
    """Replace AND clauses with multiple consecutive selects."""

    def fire(self, op):
        if not isinstance(op, algebra.Select):
            return op

        conjuncs = expression.extract_conjuncs(op.condition)
        assert conjuncs  # Must be at least 1

        # Normalize named references to integer indexes
        scheme = op.scheme()
        conjuncs = [expression.to_unnamed_recursive(c, scheme)
                    for c in conjuncs]

        op.condition = conjuncs[0]
        op.has_been_pushed = False
        for conjunc in conjuncs[1:]:
            op = algebra.Select(conjunc, op)
            op.has_been_pushed = False
        return op

    def __str__(self):
        return "Select => Select, Select"


class MergeSelects(rules.Rule):
    """Merge consecutive Selects into a single conjunctive selection."""
    def fire(self, op):
        if not isinstance(op, algebra.Select):
            return op

        while isinstance(op.input, algebra.Select):
            conjunc = expression.AND(op.condition, op.input.condition)
            op = algebra.Select(conjunc, op.input.input)

        return op

    def __str__(self):
        return "Select, Select => Select"


class ProjectToDistinctColumnSelect(rules.Rule):
    def fire(self, expr):
        # If not a Project, who cares?
        if not isinstance(expr, algebra.Project):
            return expr

        mappings = [(None, x) for x in expr.columnlist]
        colSelect = algebra.Apply(mappings, expr.input)
        # TODO(dhalperi) the distinct logic is broken because we don't have a
        # locality-aware optimizer. For now, don't insert Distinct for a
        # logical project. This is BROKEN.
        # distinct = algebra.Distinct(colSelect)
        # return distinct
        return colSelect


class SimpleGroupBy(rules.Rule):
    # A "Simple" GroupBy is one that has only AttributeRefs as its grouping
    # fields, and only AggregateExpression(AttributeRef) as its aggregate
    # fields.
    #
    # Even AggregateExpression(Literal) is more complicated than Myria's
    # GroupBy wants to handle. Thus we will insert Apply before a GroupBy to
    # take all the "Complex" expressions away.

    def fire(self, expr):
        if not isinstance(expr, algebra.GroupBy):
            return expr

        child_scheme = expr.input.scheme()

        # A simple grouping expression is an AttributeRef
        def is_simple_grp_expr(grp):
            return isinstance(grp, expression.AttributeRef)

        complex_grp_exprs = [(i, grp)
                             for (i, grp) in enumerate(expr.grouping_list)
                             if not is_simple_grp_expr(grp)]

        # A simple aggregate expression is an aggregate whose input is an
        # AttributeRef
        def is_simple_agg_expr(agg):
            return isinstance(agg, expression.COUNTALL) or \
                (isinstance(agg, expression.UnaryOperator) and
                 isinstance(agg, expression.AggregateExpression) and
                 isinstance(agg.input, expression.AttributeRef))

        complex_agg_exprs = [agg for agg in expr.aggregate_list
                             if not is_simple_agg_expr(agg)]

        # There are no complicated expressions, we're okay with the existing
        # GroupBy.
        if not complex_grp_exprs and not complex_agg_exprs:
            return expr

        # Construct the Apply we're going to stick before the GroupBy

        # First: copy every column from the input verbatim
        mappings = [(None, expression.UnnamedAttributeRef(i))
                    for i in range(len(child_scheme))]

        # Next: move the complex grouping expressions into the Apply, replace
        # with simple refs
        for i, grp_expr in complex_grp_exprs:
            mappings.append((None, grp_expr))
            expr.grouping_list[i] = \
                expression.UnnamedAttributeRef(len(mappings) - 1)

        # Finally: move the complex aggregate expressions into the Apply,
        # replace with simple refs
        for agg_expr in complex_agg_exprs:
            mappings.append((None, agg_expr.input))
            agg_expr.input = \
                expression.UnnamedAttributeRef(len(mappings) - 1)

        # Construct and prepend the new Apply
        new_apply = algebra.Apply(mappings, expr.input)
        expr.input = new_apply

        # Don't overwrite expr.grouping_list or expr.aggregate_list, instead we
        # are mutating the objects it contains when we modify grp_expr or
        # agg_expr in the above for loops.
        return expr


def is_column_equality_comparison(cond):
    """Return a tuple of column indexes if the condition is an equality test.
    """

    if isinstance(cond, expression.EQ) and \
       isinstance(cond.left, expression.UnnamedAttributeRef) and \
       isinstance(cond.right, expression.UnnamedAttributeRef):
        return (cond.left.position, cond.right.position)
    else:
        return None


class PushSelects(rules.Rule):
    """Push selections."""

    @staticmethod
    def descend_tree(op, cond):
        """Recursively push a selection condition down a tree of operators.

        :param op: The root of an operator tree
        :type op: raco.algebra.Operator
        :type cond: The selection condition
        :type cond: raco.expression.expression

        :return: A (possibly modified) operator.
        """

        if isinstance(op, algebra.Select):
            # Keep pushing; selects are commutative
            op.input = PushSelects.descend_tree(op.input, cond)
            return op
        elif isinstance(op, algebra.CompositeBinaryOperator):
            # Joins and cross-products; consider conversion to an equijoin
            left_len = len(op.left.scheme())
            accessed = expression.accessed_columns(cond)
            in_left = [col < left_len for col in accessed]
            if all(in_left):
                # Push the select into the left sub-tree.
                op.left = PushSelects.descend_tree(op.left, cond)
                return op
            elif not any(in_left):
                # Push into right subtree; rebase column indexes
                expression.rebase_expr(cond, left_len)
                op.right = PushSelects.descend_tree(op.right, cond)
                return op
            else:
                # Selection includes both children; attempt to create an
                # equijoin condition
                cols = is_column_equality_comparison(cond)
                if cols:
                    return op.add_equijoin_condition(cols[0], cols[1])

        # Can't push any more: instantiate the selection
        new_op = algebra.Select(cond, op)
        new_op.has_been_pushed = True
        return new_op

    def fire(self, op):
        if not isinstance(op, algebra.Select):
            return op
        if op.has_been_pushed:
            return op

        new_op = PushSelects.descend_tree(op.input, op.condition)

        # The new root may also be a select, so fire the rule recursively
        return self.fire(new_op)

    def __str__(self):
        return "Select, Cross/Join => Join"


class RemoveTrivialSequences(rules.Rule):
    def fire(self, expr):
        if not isinstance(expr, algebra.Sequence):
            return expr

        if len(expr.args) == 1:
            return expr.args[0]
        else:
            return expr


class MyriaAlgebra(object):
    language = MyriaLanguage

    operators = [
        MyriaSymmetricHashJoin,
        MyriaSelect,
        MyriaScan,
        MyriaStore
    ]

    fragment_leaves = (
        MyriaShuffleConsumer,
        MyriaCollectConsumer,
        MyriaBroadcastConsumer,
        MyriaScan,
        MyriaScanTemp
    )

    rules = [
        RemoveTrivialSequences(),

        SimpleGroupBy(),

        # These rules form a logical group; PushSelects assumes that
        # AND clauses have been broken apart into multiple selections.
        SplitSelects(),
        PushSelects(),
        MergeSelects(),

        rules.ProjectingJoin(),
        rules.JoinToProjectingJoin(),
        ShuffleBeforeJoin(),
        BroadcastBeforeCross(),
        DistributedGroupBy(),
        ProjectToDistinctColumnSelect(),
        rules.OneToOne(algebra.CrossProduct, MyriaCrossProduct),
        rules.OneToOne(algebra.Store, MyriaStore),
        rules.OneToOne(algebra.StoreTemp, MyriaStoreTemp),
        rules.OneToOne(algebra.StatefulApply, MyriaStatefulApply),
        rules.OneToOne(algebra.Apply, MyriaApply),
        rules.OneToOne(algebra.Select, MyriaSelect),
        rules.OneToOne(algebra.GroupBy, MyriaGroupBy),
        rules.OneToOne(algebra.Distinct, MyriaDupElim),
        rules.OneToOne(algebra.Shuffle, MyriaShuffle),
        rules.OneToOne(algebra.Collect, MyriaCollect),
        rules.OneToOne(algebra.ProjectingJoin, MyriaSymmetricHashJoin),
        rules.OneToOne(algebra.Scan, MyriaScan),
        rules.OneToOne(algebra.ScanTemp, MyriaScanTemp),
        rules.OneToOne(algebra.SingletonRelation, MyriaSingleton),
        rules.OneToOne(algebra.EmptyRelation, MyriaEmptyRelation),
        rules.OneToOne(algebra.UnionAll, MyriaUnionAll),
        BreakShuffle(),
        BreakCollect(),
        BreakBroadcast(),
    ]


def apply_schema_recursive(operator, catalog):
    """Given a catalog, which has a function get_scheme(relation_key) to map
    a relation name to its scheme, update the schema for all scan operations
    that scan relations in the map."""

    # We found a scan, let's fill in its scheme
    if isinstance(operator, MyriaScan) or isinstance(operator, MyriaScanTemp):

        if isinstance(operator, MyriaScan):
            rel_key = operator.relation_key
            rel_scheme = catalog.get_scheme(rel_key)
        elif isinstance(operator, MyriaScanTemp):
            rel_key = RelationKey.from_string(operator.name)
            rel_scheme = catalog.get_scheme(rel_key)

        if rel_scheme:
            # The Catalog has an entry for this relation
            if len(operator.scheme()) != len(rel_scheme):
                s = "scheme for %s (%d cols) does not match the catalog (%d cols)" % (rel_key, len(operator._scheme), len(rel_scheme))  # noqa
                raise ValueError(s)
            operator._scheme = rel_scheme
        else:
            # The specified relation is not in the Catalog; replace its
            # scheme's types with "unknown".
            old_sch = operator.scheme()
            new_sch = [(old_sch.getName(i), "unknown")
                       for i in range(len(old_sch))]
            operator._scheme = Scheme(new_sch)

    # Recurse through all children
    for child in operator.children():
        apply_schema_recursive(child, catalog)

    # Done
    return


class EmptyCatalog(object):
    @staticmethod
    def get_scheme(relation_name):
        return None


class SymbolFactory(object):
    def __init__(self):
        self.count = 0

    def alloc(self):
        ret = "V{0}".format(self.count)
        self.count += 1
        return ret

    def getter(self):
        return lambda: self.alloc()


def compile_to_json(raw_query, logical_plan, physical_plan, catalog=None):
    """This function compiles a logical RA plan to the JSON suitable for
    submission to the Myria REST API server."""

    # raw_query must be a string
    if not isinstance(raw_query, basestring):
        raise ValueError("raw query must be a string")

    # No catalog supplied; create the empty catalog
    if catalog is None:
        catalog = EmptyCatalog()

    # Some plans may just be an operator, others may be a list of operators
    if isinstance(physical_plan, algebra.Operator):
        physical_plan = [(None, physical_plan)]

    for (label, root_op) in physical_plan:
        apply_schema_recursive(root_op, catalog)

    # A dictionary mapping each object to a unique, object-dependent symbol.
    # Since we want this to be truly unique for each object instance, even if
    # two objects are equal, we use id(obj) as the key.
    symbol_factory = SymbolFactory()
    syms = defaultdict(symbol_factory.getter())

    def one_fragment(rootOp):
        """Given an operator that is the root of a query fragment/plan, extract
        the operators in the fragment. Assembles a list cur_frag of the
        operators in the current fragment, in preorder from the root.

        This operator also assembles a queue of the discovered roots of later
        fragments, e.g., when there is a ShuffleProducer below. The list of
        operators that should be treated as fragment leaves is given by
        MyriaAlgebra.fragment_leaves. """

        # The current fragment starts with the current root
        cur_frag = [rootOp]
        # Initially, there are no new roots discovered below leaves of this
        # fragment.
        queue = []
        if isinstance(rootOp, MyriaAlgebra.fragment_leaves):
            # The current root operator is a fragment leaf, such as a
            # ShuffleProducer. Append its children to the queue of new roots.
            for child in rootOp.children():
                queue.append(child)
        else:
            # Otherwise, the children belong in this fragment. Recursively go
            # discover their fragments, including the queue of roots below
            # their children.
            for child in rootOp.children():
                (child_frag, child_queue) = one_fragment(child)
                # Add their fragment onto this fragment
                cur_frag += child_frag
                # Add their roots-of-next-fragments into our queue
                queue += child_queue
        return (cur_frag, queue)

    def fragments(rootOp):
        """Given the root of a query plan, recursively determine all the
        fragments in it."""
        # The queue of fragment roots. Initially, just the root of this query
        queue = [rootOp]
        ret = []
        while len(queue) > 0:
            # Get the next fragment root
            rootOp = queue.pop(0)
            # .. recursively learn the entire fragment, and any newly
            # discovered roots.
            (op_frag, op_queue) = one_fragment(rootOp)
            # .. Myria JSON expects the fragment operators in reverse order,
            # i.e., root at the bottom.
            ret.append(reversed(op_frag))
            # .. and collect the newly discovered fragment roots.
            queue.extend(op_queue)
        return ret

    def call_compile_me(op):
        "A shortcut to call the operator's compile_me function."
        opsym = syms[id(op)]
        childsyms = [syms[id(child)] for child in op.children()]
        if isinstance(op, algebra.ZeroaryOperator):
            return op.compileme(opsym)
        if isinstance(op, algebra.UnaryOperator):
            return op.compileme(opsym, childsyms[0])
        if isinstance(op, algebra.BinaryOperator):
            return op.compileme(opsym, childsyms[0], childsyms[1])
        if isinstance(op, algebra.NaryOperator):
            return op.compileme(opsym, childsyms)
        raise NotImplementedError("unable to handle operator of type " + type(op))  # noqa

    # The actual code. all_frags collects up the fragments.
    all_frags = []
    # For each IDB, generate a plan that assembles all its fragments and stores
    # them back to a relation named (label).
    for (label, rootOp) in physical_plan:

        # If the root operator is not a Store-type, we need to add one at the
        # top. We actually do this later, but we want to allocate the new
        # operator's label first
        if not isinstance(rootOp, (algebra.Store, algebra.StoreTemp)):
            store_label = symbol_factory.alloc()

        # Sometimes the root operator is not labeled, usually because we were
        # lazy when submitting a manual plan. In this case, generate a new
        # label.
        if not label:
            label = syms[id(rootOp)]

        if not isinstance(rootOp, (algebra.Store, algebra.StoreTemp)):
            # Here we actually create the Store that goes at the root
            frag_root = MyriaStore(plan=rootOp,
                                   relation_key=RelationKey.from_string(label))
            label = store_label
            del store_label                 # Aggressive bug detection
        else:
            frag_root = rootOp

        # Make sure the root is in the symbol dictionary, but rather than using
        # a generated symbol use the IDB label.
        syms[id(frag_root)] = label
        # Determine the fragments.
        frags = fragments(frag_root)
        # Build the fragments.
        all_frags.extend([{'operators': [call_compile_me(op) for op in frag]}
                          for frag in frags])
        # Clear out the symbol dictionary for the next IDB.
        syms.clear()

    # Assemble all the fragments into a single JSON query plan
    query = {
        'fragments': all_frags,
        'rawDatalog': raw_query,
        'logicalRa': str(logical_plan)
    }
    return query
