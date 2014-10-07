import itertools
import logging
from collections import defaultdict, deque
from operator import mul
from sqlalchemy.dialects import postgresql

from raco import algebra, expression, rules
from raco.catalog import Catalog
from raco.language import Language, Algebra
from raco.language.sql.catalog import SQLCatalog
from raco.expression import UnnamedAttributeRef
from raco.expression.aggregate import (rebase_local_aggregate_output,
                                       rebase_finalizer)
from raco.expression.statevar import *
from raco.datastructure.UnionFind import UnionFind
from raco import types

LOGGER = logging.getLogger(__name__)


def scheme_to_schema(s):
    if s:
        names, descrs = zip(*s.asdict.items())
        names = ["%s" % n for n in names]
        types_ = [r[1] for r in descrs]
    else:
        names = []
        types_ = []
    return {"columnTypes": types_, "columnNames": names}


def compile_expr(op, child_scheme, state_scheme):
    ####
    # Put special handling at the top!
    ####
    if isinstance(op, expression.NumericLiteral):
        if type(op.value) == int:
            myria_type = types.LONG_TYPE
        elif type(op.value) == float:
            myria_type = types.DOUBLE_TYPE
        else:
            raise NotImplementedError("Compiling NumericLiteral {} of type {}"
                                      .format(op, type(op.value)))

        return {
            'type': 'CONSTANT',
            'value': str(op.value),
            'valueType': myria_type
        }
    elif isinstance(op, expression.StringLiteral):
        return {
            'type': 'CONSTANT',
            'value': str(op.value),
            'valueType': types.STRING_TYPE
        }
    elif isinstance(op, expression.BooleanLiteral):
        return {
            'type': 'CONSTANT',
            'value': bool(op.value),
            'valueType': 'BOOLEAN_TYPE'
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
    elif isinstance(op, expression.CAST):
        return {
            'type': 'CAST',
            'left': compile_expr(op.input, child_scheme, state_scheme),
            'right': {
                'type': 'TYPE',
                'outputType': op._type
            }
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
    elif isinstance(op, expression.ZeroaryOperator):
        return {
            'type': op.opname(),
        }
    elif isinstance(op, expression.NaryOperator):
        children = []
        for operand in op.operands:
            children.append(compile_expr(operand, child_scheme, state_scheme))
        return {
            'type': op.opname(),
            'children': children
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


class MyriaOperator(object):
    language = MyriaLanguage


def relation_key_to_json(relation_key):
    return {"userName": relation_key.user,
            "programName": relation_key.program,
            "relationName": relation_key.relation}


class MyriaScan(algebra.Scan, MyriaOperator):
    def compileme(self):
        return {
            "opType": "TableScan",
            "relationKey": relation_key_to_json(self.relation_key),
        }


class MyriaScanTemp(algebra.ScanTemp, MyriaOperator):
    def compileme(self):
        return {
            "opType": "TempTableScan",
            "table": self.name,
        }


class MyriaUnionAll(algebra.UnionAll, MyriaOperator):
    def compileme(self, leftid, rightid):
        return {
            "opType": "UnionAll",
            "argChildren": [leftid, rightid]
        }


class MyriaDifference(algebra.Difference, MyriaOperator):
    def compileme(self, leftid, rightid):
        return {
            "opType": "Difference",
            "argChild1": leftid,
            "argChild2": rightid,
        }


class MyriaSingleton(algebra.SingletonRelation, MyriaOperator):
    def compileme(self):
        return {
            "opType": "Singleton",
        }


class MyriaEmptyRelation(algebra.EmptyRelation, MyriaOperator):
    def compileme(self):
        return {
            "opType": "Empty",
            'schema': scheme_to_schema(self.scheme())
        }


class MyriaSelect(algebra.Select, MyriaOperator):
    def compileme(self, inputid):
        pred = compile_expr(self.condition, self.scheme(), None)
        return {
            "opType": "Filter",
            "argChild": inputid,
            "argPredicate": {
                "rootExpressionOperator": pred
            }
        }


class MyriaCrossProduct(algebra.CrossProduct, MyriaOperator):
    def compileme(self, leftid, rightid):
        column_names = [name for (name, _) in self.scheme()]
        allleft = [i.position for i in self.left.scheme().ascolumnlist()]
        allright = [i.position for i in self.right.scheme().ascolumnlist()]
        return {
            "opType": "SymmetricHashJoin",
            "argColumnNames": column_names,
            "argChild1": leftid,
            "argChild2": rightid,
            "argColumns1": [],
            "argColumns2": [],
            "argSelect1": allleft,
            "argSelect2": allright
        }


class MyriaStore(algebra.Store, MyriaOperator):
    def compileme(self, inputid):
        return {
            "opType": "DbInsert",
            "relationKey": relation_key_to_json(self.relation_key),
            "argOverwriteTable": True,
            "argChild": inputid,
        }


class MyriaStoreTemp(algebra.StoreTemp, MyriaOperator):
    def compileme(self, inputid):
        return {
            "opType": "TempInsert",
            "table": self.name,
            "argOverwriteTable": True,
            "argChild": inputid,
        }


class MyriaAppendTemp(algebra.AppendTemp, MyriaOperator):
    def compileme(self, inputid):
        return {
            "opType": "TempInsert",
            "table": self.name,
            "argOverwriteTable": False,
            "argChild": inputid,
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


def convert_nary_conditions(conditions, schemes):
    """Convert an NaryJoin map from global column index to local"""
    attr_map = {}   # map of global attribute to local column index
    count = 0
    for i, scheme in enumerate(schemes):
        for j, attr in enumerate(scheme.ascolumnlist()):
            attr_map[count] = [i, j]
            count += 1
    new_conditions = []   # arrays of [child_index, column_index]
    for join_cond in conditions:
        new_join_cond = []
        for attr in join_cond:
            new_join_cond.append(attr_map[attr.position])
        new_conditions.append(new_join_cond)
    return new_conditions


class MyriaSymmetricHashJoin(algebra.ProjectingJoin, MyriaOperator):
    def compileme(self, leftid, rightid):
        """Compile the operator to a sequence of json operators"""

        left_len = len(self.left.scheme())
        combined = self.left.scheme() + self.right.scheme()
        leftcols, rightcols = convertcondition(self.condition,
                                               left_len,
                                               combined)

        if self.output_columns is None:
            self.output_columns = self.scheme().ascolumnlist()
        column_names = [name for (name, _) in self.scheme()]
        pos = [i.get_position(combined) for i in self.output_columns]
        side = [p >= left_len for p in pos]
        assert sorted(side) == side, \
            "MyriaSymmetricHashJoin always emits left columns first"
        allleft = [i for i in pos if i < left_len]
        allright = [i - left_len for i in pos if i >= left_len]

        join = {
            "opType": "SymmetricHashJoin",
            "argColumnNames": column_names,
            "argChild1": "%s" % leftid,
            "argColumns1": leftcols,
            "argChild2": "%s" % rightid,
            "argColumns2": rightcols,
            "argSelect1": allleft,
            "argSelect2": allright
        }

        return join


class MyriaLeapFrogJoin(algebra.NaryJoin, MyriaOperator):

    def compileme(self, *args):
        def convert_join_cond(pos_to_rel_col, cond, scheme):
            join_col_pos = [c.get_position(scheme) for c in cond]
            return [pos_to_rel_col[p] for p in join_col_pos]
        # map a output column to its origin
        rel_of_pos = {}     # pos => [rel_idx, field_idx]
        schemes = [c.scheme().ascolumnlist() for c in self.children()]
        pos = 0
        combined = []
        for rel_idx, scheme in enumerate(schemes):
            combined.extend(scheme)
            for field_idx in xrange(len(scheme)):
                rel_of_pos[pos] = [rel_idx, field_idx]
                pos += 1
        # build column names
        if self.output_columns is None:
            self.output_columns = self.scheme().ascolumnlist()
        column_names = [name for (name, _) in self.scheme()]
        # get rel_idx and field_idx of select columns
        out_pos_list = [
            i.get_position(combined) for i in list(self.output_columns)]
        output_fields = [rel_of_pos[p] for p in out_pos_list]
        join_fields = [
            convert_join_cond(rel_of_pos, cond, combined)
            for cond in self.conditions]
        return {
            "opType": "LeapFrogJoin",
            "joinFieldMapping": join_fields,
            "argColumnNames": column_names,
            "outputFieldMapping": output_fields,
            "argChildren": args
        }


class MyriaGroupBy(algebra.GroupBy, MyriaOperator):
    @staticmethod
    def agg_mapping(agg_expr):
        """Maps a BuiltinAggregateExpression to a Myria string constant
        representing the corresponding aggregate operation."""
        if isinstance(agg_expr, expression.MAX):
            return "MAX"
        elif isinstance(agg_expr, expression.MIN):
            return "MIN"
        elif isinstance(agg_expr, expression.SUM):
            return "SUM"
        elif isinstance(agg_expr, expression.AVG):
            return "AVG"
        elif isinstance(agg_expr, expression.STDEV):
            return "STDEV"
        raise NotImplementedError("MyriaGroupBy.agg_mapping({})".format(
            type(agg_expr)))

    @staticmethod
    def compile_builtin_agg(agg, child_scheme):
        assert isinstance(agg, expression.BuiltinAggregateExpression)
        if isinstance(agg, expression.COUNTALL):
            return {"type": "CountAll"}

        assert isinstance(agg, expression.UnaryOperator)
        column = expression.toUnnamed(agg.input, child_scheme).position
        return {"type": "SingleColumn",
                "aggOps": [MyriaGroupBy.agg_mapping(agg)],
                "column": column}

    def compileme(self, inputid):
        child_scheme = self.input.scheme()
        group_fields = [expression.toUnnamed(ref, child_scheme)
                        for ref in self.grouping_list]

        built_ins = [agg_expr for agg_expr in self.aggregate_list
                     if isinstance(agg_expr,
                                   expression.BuiltinAggregateExpression)]

        aggregators = [MyriaGroupBy.compile_builtin_agg(agg_expr, child_scheme)
                       for agg_expr in built_ins]

        assert all(aggregators[i] != aggregators[j]
                   for i in range(len(aggregators))
                   for j in range(len(aggregators))
                   if i < j)

        udas = [agg_expr for agg_expr in self.aggregate_list
                if isinstance(agg_expr, expression.UdaAggregateExpression)]
        assert len(udas) + len(built_ins) == len(self.aggregate_list)

        if udas:
            inits = [compile_mapping(e, None, None) for e in self.inits]
            updates = [compile_mapping(e, child_scheme, self.state_scheme)
                       for e in self.updaters]
            emitters = [compile_mapping(("uda{i}".format(i=i),
                                         e.input),
                                        None, self.state_scheme)
                        for i, e in enumerate(udas)]
            aggregators.append({
                "type": "UserDefined",
                "initializers": inits,
                "updaters": updates,
                "emitters": emitters
            })

        ret = {
            "argChild": inputid,
            "aggregators": aggregators,
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


class MyriaInMemoryOrderBy(algebra.OrderBy, MyriaOperator):

    def compileme(self, inputsym):
        return {
            "opType": "InMemoryOrderBy",
            "argChild": inputsym,
            "argSortColumns": self.sort_columns,
            "argAscending": self.ascending
        }


class MyriaShuffle(algebra.Shuffle, MyriaOperator):
    """Represents a simple shuffle operator"""

    def compileme(self, inputid):
        raise NotImplementedError('shouldn''t ever get here, should be turned into SP-SC pair')  # noqa


class MyriaCollect(algebra.Collect, MyriaOperator):
    """Represents a simple collect operator"""

    def compileme(self, inputid):
        raise NotImplementedError('shouldn''t ever get here, should be turned into CP-CC pair')  # noqa


class MyriaDupElim(algebra.Distinct, MyriaOperator):
    """Represents duplicate elimination"""

    def compileme(self, inputid):
        return {
            "opType": "DupElim",
            "argChild": inputid,
        }


class MyriaApply(algebra.Apply, MyriaOperator):
    """Represents a simple apply operator"""

    def compileme(self, inputid):
        child_scheme = self.input.scheme()
        emitters = [compile_mapping(x, child_scheme, None)
                    for x in self.emitters]
        return {
            'opType': 'Apply',
            'argChild': inputid,
            'emitExpressions': emitters
        }


class MyriaStatefulApply(algebra.StatefulApply, MyriaOperator):
    """Represents a stateful apply operator"""

    def compileme(self, inputid):
        child_scheme = self.input.scheme()
        state_scheme = self.state_scheme
        comp_map = lambda x: compile_mapping(x, child_scheme, state_scheme)
        emitters = [comp_map(x) for x in self.emitters]
        inits = [comp_map(x) for x in self.inits]
        updaters = [comp_map(x) for x in self.updaters]
        return {
            'opType': 'StatefulApply',
            'argChild': inputid,
            'emitExpressions': emitters,
            'initializerExpressions': inits,
            'updaterExpressions': updaters
        }


class MyriaBroadcastProducer(algebra.UnaryOperator, MyriaOperator):
    """A Myria BroadcastProducer"""

    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)

    def num_tuples(self):
        return self.input.num_tuples()

    def shortStr(self):
        return "%s" % self.opname()

    def compileme(self, inputid):
        return {
            "opType": "BroadcastProducer",
            "argChild": inputid,
        }


class MyriaBroadcastConsumer(algebra.UnaryOperator, MyriaOperator):
    """A Myria BroadcastConsumer"""

    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)

    def num_tuples(self):
        return self.input.num_tuples()

    def shortStr(self):
        return "%s" % self.opname()

    def compileme(self, inputid):
        return {
            'opType': 'BroadcastConsumer',
            'argOperatorId': inputid
        }


class MyriaShuffleProducer(algebra.UnaryOperator, MyriaOperator):
    """A Myria ShuffleProducer"""

    def __init__(self, input, hash_columns):
        algebra.UnaryOperator.__init__(self, input)
        self.hash_columns = hash_columns

    def shortStr(self):
        hash_string = ','.join([str(x) for x in self.hash_columns])
        return "%s(h(%s))" % (self.opname(), hash_string)

    def __repr__(self):
        return "{op}({inp!r}, {hc!r})".format(op=self.opname(), inp=self.input,
                                              hc=self.hash_columns)

    def num_tuples(self):
        return self.input.num_tuples()

    def compileme(self, inputid):
        if len(self.hash_columns) == 1:
            pf = {
                "type": "SingleFieldHash",
                "index": self.hash_columns[0].position
            }
        else:
            pf = {
                "type": "MultiFieldHash",
                "indexes": [x.position for x in self.hash_columns]
            }

        return {
            "opType": "ShuffleProducer",
            "argChild": inputid,
            "argPf": pf
        }


class MyriaShuffleConsumer(algebra.UnaryOperator, MyriaOperator):
    """A Myria ShuffleConsumer"""

    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)

    def num_tuples(self):
        return self.input.num_tuples()

    def shortStr(self):
        return "%s" % self.opname()

    def compileme(self, inputid):
        return {
            'opType': 'ShuffleConsumer',
            'argOperatorId': inputid
        }


class MyriaCollectProducer(algebra.UnaryOperator, MyriaOperator):
    """A Myria CollectProducer"""

    def __init__(self, input, server):
        algebra.UnaryOperator.__init__(self, input)
        self.server = server

    def num_tuples(self):
        return self.input.num_tuples()

    def shortStr(self):
        return "%s(@%s)" % (self.opname(), self.server)

    def compileme(self, inputid):
        return {
            "opType": "CollectProducer",
            "argChild": inputid,
        }

    def __repr__(self):
        return "{op}({inp!r}, {svr!r})".format(op=self.opname(),
                                               inp=self.input,
                                               svr=self.server)


class MyriaCollectConsumer(algebra.UnaryOperator, MyriaOperator):
    """A Myria CollectConsumer"""

    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)

    def num_tuples(self):
        return self.input.num_tuples()

    def shortStr(self):
        return "%s" % self.opname()

    def compileme(self, inputid):
        return {
            'opType': 'CollectConsumer',
            'argOperatorId': inputid
        }


class MyriaHyperShuffle(algebra.HyperCubeShuffle, MyriaOperator):
    """Represents a HyperShuffle shuffle operator"""
    def compileme(self, inputsym):
        raise NotImplementedError('shouldn''t ever get here, should be turned into HCSP-HCSC pair')  # noqa


class MyriaHyperShuffleProducer(algebra.UnaryOperator, MyriaOperator):
    """A Myria HyperShuffleProducer"""
    def __init__(self, input, hashed_columns,
                 hyper_cube_dims, mapped_hc_dims, cell_partition):
        algebra.UnaryOperator.__init__(self, input)
        self.hashed_columns = hashed_columns
        self.mapped_hc_dimensions = mapped_hc_dims
        self.hyper_cube_dimensions = hyper_cube_dims
        self.cell_partition = cell_partition

    def num_tuples(self):
        return self.input.num_tuples()

    def shortStr(self):
        mapping = {i: '*' for i in range(len(self.hyper_cube_dimensions))}
        mapping.update({h: 'h({col})'.format(col=i)
                        for i, h in zip(self.hashed_columns,
                                        self.mapped_hc_dimensions)})
        hash_string = ','.join(s for m, s in sorted(mapping.items()))
        return "%s(%s)" % (self.opname(), hash_string)

    def compileme(self, inputsym):
        return {
            "opType": "HyperShuffleProducer",
            "hashedColumns": list(self.hashed_columns),
            "mappedHCDimensions": list(self.mapped_hc_dimensions),
            "hyperCubeDimensions": list(self.hyper_cube_dimensions),
            "cellPartition": self.cell_partition,
            "argChild": inputsym
        }


class MyriaHyperShuffleConsumer(algebra.UnaryOperator, MyriaOperator):
    """A Myria HyperShuffleConsumer"""
    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)

    def num_tuples(self):
        return self.input.num_tuples()

    def shortStr(self):
        return "%s" % self.opname()

    def compileme(self, inputsym):
        return {
            "opType": "HyperShuffleConsumer",
            "argOperatorId": inputsym
        }


class MyriaQueryScan(algebra.ZeroaryOperator, MyriaOperator):
    """A Myria Query Scan"""
    def __init__(self, sql, scheme, num_tuples=algebra.DEFAULT_CARDINALITY):
        self.sql = str(sql)
        self._scheme = scheme
        self._num_tuples = num_tuples

    def __repr__(self):
        return ("{op}({sql!r}, {sch!r}, {nt!r})"
                .format(op=self.opname(), sql=self.sql,
                        sch=self._scheme, nt=self._num_tuples))

    def num_tuples(self):
        return self._num_tuples

    def shortStr(self):
        return "MyriaQueryScan({sql!r})".format(sql=self.sql)

    def scheme(self):
        return self._scheme

    def compileme(self):
        return {
            "opType": "DbQueryScan",
            "sql": self.sql,
            "schema": scheme_to_schema(self._scheme)
        }


class BreakShuffle(rules.Rule):
    def fire(self, expr):
        if not isinstance(expr, MyriaShuffle):
            return expr

        producer = MyriaShuffleProducer(expr.input, expr.columnlist)
        consumer = MyriaShuffleConsumer(producer)
        return consumer


class BreakHyperCubeShuffle(rules.Rule):
    def fire(self, expr):
        """
        self.hashed_columns = hashed_columns
        self.mapped_hc_dimensions = mapped_hc_dims
        self.hyper_cube_dimensions = hyper_cube_dims
        self.cell_partition = cell_partition
        """
        if not isinstance(expr, MyriaHyperShuffle):
            return expr
        producer = MyriaHyperShuffleProducer(
            expr.input, expr.hashed_columns, expr.hyper_cube_dimensions,
            expr.mapped_hc_dimensions, expr.cell_partition)
        consumer = MyriaHyperShuffleConsumer(producer)
        return consumer


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


def check_shuffle_xor(exp):
    """Enforce that neither or both inputs to a binary op are shuffled.

    Return True if the arguments are shuffled; False if they are not;
    or raise a ValueError on xor failure.

    Note that we assume that inputs are shuffled in a compatible way.
    """
    left_shuffle = isinstance(exp.left, algebra.Shuffle)
    right_shuffle = isinstance(exp.right, algebra.Shuffle)

    if left_shuffle and right_shuffle:
        return True
    if left_shuffle or right_shuffle:
        raise ValueError("Must shuffle on both inputs of %s" % exp)
    return False


class ShuffleBeforeSetop(rules.Rule):
    def fire(self, exp):
        if not isinstance(exp, (algebra.Difference, algebra.Intersection)):
            return exp

        def shuffle_after(op):
            cols = [expression.UnnamedAttributeRef(i)
                    for i in range(len(op.scheme()))]
            return algebra.Shuffle(child=op, columnlist=cols)

        if not check_shuffle_xor(exp):
            exp.left = shuffle_after(exp.left)
            exp.right = shuffle_after(exp.right)
        return exp


class ShuffleBeforeJoin(rules.Rule):
    def fire(self, expr):
        # If not a join, who cares?
        if not isinstance(expr, algebra.Join):
            return expr

        # If both have shuffles already, who cares?
        if check_shuffle_xor(expr):
            return expr

        # Figure out which columns go in the shuffle
        left_cols, right_cols = \
            convertcondition(expr.condition,
                             len(expr.left.scheme()),
                             expr.left.scheme() + expr.right.scheme())

        # Left shuffle
        left_cols = [expression.UnnamedAttributeRef(i)
                     for i in left_cols]
        left_shuffle = algebra.Shuffle(expr.left, left_cols)
        # Right shuffle
        right_cols = [expression.UnnamedAttributeRef(i)
                      for i in right_cols]
        right_shuffle = algebra.Shuffle(expr.right, right_cols)

        # Construct the object!
        assert isinstance(expr, algebra.ProjectingJoin)
        if isinstance(expr, algebra.ProjectingJoin):
            return algebra.ProjectingJoin(expr.condition,
                                          left_shuffle, right_shuffle,
                                          expr.output_columns)


class HCShuffleBeforeNaryJoin(rules.Rule):
    def __init__(self, catalog):
        assert isinstance(catalog, Catalog)
        self.catalog = catalog

    @staticmethod
    def reversed_index(child_schemes, conditions):
        """Return the reversed index of join conditions. The reverse index
           specify for each column on each relation, which hypercube dimension
           it is mapped to, -1 means this columns is not in the hyper cube
           (not joined).

        Keyword arguments:
        child_schemes -- schemes of children.
        conditions -- join conditions.
        """
        # make it -1 first
        r_index = [[-1] * len(scheme) for scheme in child_schemes]
        for i, jf_list in enumerate(conditions):
            for jf in jf_list:
                r_index[jf[0]][jf[1]] = i
        return r_index

    @staticmethod
    def workload(dim_sizes, child_sizes, r_index):
        """Compute the workload given a hyper cube size assignment"""
        load = 0.0
        for i, size in enumerate(child_sizes):
            # compute subcube sizes
            scale = 1
            for index in r_index[i]:
                if index != -1:
                    scale = scale * dim_sizes[index]
            # add load per server by child i
            load += float(child_sizes[i]) / float(scale)
        return load

    @staticmethod
    def get_hyper_cube_dim_size(num_server, child_sizes,
                                conditions, r_index):
        """Find the optimal hyper cube dimension sizes using BFS.

        Keyword arguments:
        num_server -- number of servers, this sets upper bound of HC cells.
        child_sizes -- cardinality of each child.
        child_schemes -- schemes of children.
        conditions -- join conditions.
        r_index -- reversed index of join conditions.
        """
        # Helper function: compute the product.
        def product(array):
            return reduce(mul, array, 1)
        # Use BFS to find the best possible assignment.
        this = HCShuffleBeforeNaryJoin
        visited = set()
        toVisit = deque()
        toVisit.append(tuple([1 for _ in conditions]))
        min_work_load = None
        while len(toVisit) > 0:
            dim_sizes = toVisit.pop()
            if ((this.workload(dim_sizes, child_sizes, r_index) <
                    min_work_load) or (min_work_load is None)):
                min_work_load = this.workload(
                    dim_sizes, child_sizes, r_index)
                opt_dim_sizes = dim_sizes
            visited.add(dim_sizes)
            for i, d in enumerate(dim_sizes):
                new_dim_sizes = (dim_sizes[0:i] +
                                 tuple([dim_sizes[i] + 1]) +
                                 dim_sizes[i + 1:])
                if (product(new_dim_sizes) <= num_server
                        and new_dim_sizes not in visited):
                    toVisit.append(new_dim_sizes)
        return opt_dim_sizes, min_work_load

    @staticmethod
    def coord_to_worker_id(coordinate, dim_sizes):
        """Convert coordinate of cell to worker id

        Keyword arguments:
        coordinate -- coordinate of hyper cube cell.
        dim_sizes -- sizes of dimensons of hyper cube.
        """
        assert len(coordinate) == len(dim_sizes)
        ret = 0
        for k, v in enumerate(coordinate):
            ret += v * reduce(mul, dim_sizes[k + 1:], 1)
        return ret

    @staticmethod
    def get_cell_partition(dim_sizes, conditions,
                           child_schemes, child_idx, hashed_columns):
        """Generate the cell_partition for a specific child.

        Keyword arguments:
        dim_sizes -- size of each dimension of the hypercube.
        conditions -- each element is an array of (child_idx, column).
        child_schemes -- schemes of children.
        child_idx -- index of this child.
        hashed_columns -- hashed columns of this child.
        """
        assert len(dim_sizes) == len(conditions)
        # make life a little bit easier
        this = HCShuffleBeforeNaryJoin
        # get reverse index
        r_index = this.reversed_index(child_schemes, conditions)
        # find which dims in hyper cube this relation is involved
        hashed_dims = [r_index[child_idx][col] for col in hashed_columns]
        assert -1 not in hashed_dims
        # group by cell according to their projection on subcube voxel
        cell_partition = defaultdict(list)
        coor_ranges = [list(range(d)) for d in dim_sizes]
        for coordinate in itertools.product(*coor_ranges):
            # project a hypercube cell to a subcube voxel
            voxel = [coordinate[dim] for dim in hashed_dims]
            cell_partition[tuple(voxel)].append(
                this.coord_to_worker_id(coordinate, dim_sizes))
        return [wid for vox, wid in sorted(cell_partition.items())]

    def fire(self, expr):
        def add_hyper_shuffle():
            """ Helper function: put a HyperCube shuffle before each child."""
            # make calling static method easier
            this = HCShuffleBeforeNaryJoin
            # get child schemes
            child_schemes = [op.scheme() for op in expr.children()]
            # convert join conditions from expressions to 2d array
            conditions = convert_nary_conditions(
                expr.conditions, child_schemes)
            # get number of servers from catalog
            num_server = self.catalog.get_num_servers()
            # get estimated cardinalities of children
            child_sizes = [child.num_tuples() for child in expr.children()]
            # get reversed index of join conditions
            r_index = this.reversed_index(child_schemes, conditions)
            # compute optimal dimension sizes
            (dim_sizes, workload) = this.get_hyper_cube_dim_size(
                num_server, child_sizes, conditions, r_index)
            # specify HyperCube shuffle to each child
            new_children = []
            for child_idx, child in enumerate(expr.children()):
                # (mapped hc dimension, column index)
                hashed_fields = [(hc_dim, i)
                                 for i, hc_dim
                                 in enumerate(r_index[child_idx])
                                 if hc_dim != -1]
                mapped_dims, hashed_columns = zip(*sorted(hashed_fields))
                # get cell partition for child i
                cell_partition = this.get_cell_partition(
                    dim_sizes, conditions, child_schemes,
                    child_idx, hashed_columns)
                # generate new children
                new_children.append(
                    algebra.HyperCubeShuffle(
                        child, hashed_columns, mapped_dims,
                        dim_sizes, cell_partition))
            # replace the children
            expr.args = new_children

        # only apply to NaryJoin
        if not isinstance(expr, algebra.NaryJoin):
            return expr
        # check if HC shuffle has been placed before
        shuffled_child = [isinstance(op, algebra.HyperCubeShuffle)
                          for op in list(expr.children())]
        if all(shuffled_child):    # already shuffled
            assert len(expr.children()) > 0
            return expr
        elif any(shuffled_child):
            raise NotImplementedError("NaryJoin is partially shuffled?")
        else:                      # add shuffle and order by
            add_hyper_shuffle()
            return expr


class OrderByBeforeNaryJoin(rules.Rule):
    def fire(self, expr):
        # if not NaryJoin, who cares?
        if not isinstance(expr, algebra.NaryJoin):
            return expr
        ordered_child = sum(
            [1 for child in expr.children()
             if isinstance(child, algebra.OrderBy)])

        # already applied
        if ordered_child == len(expr.children()):
            return expr
        elif ordered_child > 0:
            raise Exception("children are partially ordered? ")

        new_children = []
        for child in expr.children():
            # check: this rule must be applied after shuffle
            assert isinstance(child, algebra.HyperCubeShuffle)
            ascending = [True] * len(child.hashed_columns)
            new_children.append(
                algebra.OrderBy(
                    child, child.hashed_columns, ascending))
        expr.args = new_children
        return expr


class BroadcastBeforeCross(rules.Rule):
    def fire(self, expr):
        # If not a CrossProduct, who cares?
        if not isinstance(expr, algebra.CrossProduct):
            return expr

        if (isinstance(expr.left, algebra.Broadcast) or
                isinstance(expr.right, algebra.Broadcast)):
            return expr

        try:
            # By default, broadcast the smaller child
            if expr.left.num_tuples() < expr.right.num_tuples():
                expr.left = algebra.Broadcast(expr.left)
            else:
                expr.right = algebra.Broadcast(expr.right)
        except NotImplementedError, e:
            # If cardinalities unknown, broadcast the right child
            expr.right = algebra.Broadcast(expr.right)

        return expr


class DecomposeGroupBy(rules.Rule):
    """Convert a logical group by into a two-phase group by.

    The local half of the aggregate before the shuffle step, whereas the remote
    half runs after the shuffle step.

    TODO: omit this optimization if the data is already shuffled, or
    if the cardinality of the grouping keys is high.
    """

    @staticmethod
    def do_transfer(op):
        """Introduce a network transfer before a groupby operation."""

        # Get an array of position references to columns in the child scheme
        child_scheme = op.input.scheme()
        group_fields = [expression.toUnnamed(ref, child_scheme)
                        for ref in op.grouping_list]
        if len(group_fields) == 0:
            # Need to Collect all tuples at once place
            op.input = algebra.Collect(op.input)
        else:
            # Need to Shuffle
            op.input = algebra.Shuffle(op.input, group_fields)

    def fire(self, op):
        # Punt if it's not a group by or we've already converted this into an
        # an instance of MyriaGroupBy
        if op.__class__ != algebra.GroupBy:
            return op

        # Bail early if we have any non-decomposable aggregates
        if not all(x.is_decomposable() for x in op.aggregate_list):
            out_op = MyriaGroupBy()
            out_op.copy(op)
            DecomposeGroupBy.do_transfer(out_op)
            return out_op

        num_grouping_terms = len(op.grouping_list)

        local_emitters = []
        local_statemods = []
        remote_emitters = []
        remote_statemods = []
        finalizer_exprs = []

        # The starting positions for the current local, remote aggregate
        local_output_pos = num_grouping_terms
        remote_output_pos = num_grouping_terms
        requires_finalizer = False

        for agg in op.aggregate_list:
            # Multiple emit arguments can be associated with a single
            # decomposition rule; coalesce them all together.
            state = agg.get_decomposable_state()
            assert state

            ################################
            # Extract the set of emitters and statemods required for the
            # local aggregate.
            ################################

            laggs = state.get_local_emitters()
            local_emitters.extend(laggs)
            local_statemods.extend(state.get_local_statemods())

            ################################
            # Extract the set of emitters and statemods required for the
            # remote aggregate.  Remote expressions must be rebased to
            # remove instances of LocalAggregateOutput
            ################################

            raggs = state.get_remote_emitters()
            raggs = [rebase_local_aggregate_output(x, local_output_pos)
                     for x in raggs]
            remote_emitters.extend(raggs)

            rsms = state.get_remote_statemods()
            for sm in rsms:
                update_expr = rebase_local_aggregate_output(
                    sm.update_expr, local_output_pos)
                remote_statemods.append(
                    StateVar(sm.name, sm.init_expr, update_expr))

            ################################
            # Extract any required finalizers.  These must be rebased to remove
            # instances of RemoteAggregateOutput
            ################################

            finalizer = state.get_finalizer()
            if finalizer is not None:
                requires_finalizer = True
                finalizer_exprs.append(
                    rebase_finalizer(finalizer, remote_output_pos))
            else:
                for i in range(len(raggs)):
                    finalizer_exprs.append(
                        UnnamedAttributeRef(remote_output_pos + i))

            local_output_pos += len(laggs)
            remote_output_pos += len(raggs)

        ################################
        # Glue together the local and remote aggregates:
        # Local => Shuffle => Remote => (optional) Finalizer.
        ################################

        local_gb = MyriaGroupBy(op.grouping_list, local_emitters, op.input,
                                local_statemods)

        grouping_fields = [UnnamedAttributeRef(i)
                           for i in range(num_grouping_terms)]

        remote_gb = MyriaGroupBy(grouping_fields, remote_emitters, local_gb,
                                 remote_statemods)

        DecomposeGroupBy.do_transfer(remote_gb)

        if requires_finalizer:
            # Pass through grouping terms
            gmappings = [(None, UnnamedAttributeRef(i))
                         for i in range(num_grouping_terms)]
            fmappings = [(None, fx) for fx in finalizer_exprs]
            return algebra.Apply(gmappings + fmappings, remote_gb)
        return remote_gb


class AddAppendTemp(rules.Rule):
    def fire(self, op):
        if type(op) is not MyriaStoreTemp:
            return op

        child = op.input
        if type(child) is not MyriaUnionAll:
            return op

        left = child.left
        right = child.right
        rel_name = op.name

        is_scan = lambda op: type(op) is MyriaScanTemp and op.name == rel_name
        if is_scan(left) and not any(is_scan(op) for op in right.walk()):
                return MyriaAppendTemp(name=rel_name, input=right)

        elif is_scan(right) and not any(is_scan(op) for op in left.walk()):
                return MyriaAppendTemp(name=rel_name, input=left)

        return op


class PushIntoSQL(rules.Rule):
    def __init__(self, dialect=None):
        self.dialect = dialect or postgresql.dialect()

    def fire(self, expr):
        if isinstance(expr, (algebra.Scan, algebra.ScanTemp)):
            return expr
        cat = SQLCatalog()
        try:
            sql_plan = cat.get_sql(expr)
            sql_string = sql_plan.compile(dialect=self.dialect)
            sql_string.visit_bindparam = sql_string.render_literal_bindparam
            return MyriaQueryScan(sql=sql_string.process(sql_plan),
                                  scheme=expr.scheme(),
                                  num_tuples=expr.num_tuples())
        except NotImplementedError, e:
            LOGGER.warn("Error converting {plan}: {e}"
                        .format(plan=expr, e=e))
            return expr


class MergeToNaryJoin(rules.Rule):
    """Merge consecutive binary join into a single multiway join
    Note: this code assumes that the binary joins form a left deep tree
    before the merge."""
    @staticmethod
    def mergable(op):
        """Recursively checks whether an operator is mergable to NaryJoin.
        An operator will be merged to NaryJoin if its subtree contains
        only joins.
        """
        allowed_intermediate_types = (algebra.ProjectingJoin, algebra.Select)
        if issubclass(type(op), algebra.ZeroaryOperator):
            return True
        if not isinstance(op, allowed_intermediate_types):
            return False
        elif issubclass(type(op), algebra.UnaryOperator):
            return MergeToNaryJoin.mergable(op.input)
        elif issubclass(type(op), algebra.BinaryOperator):
            return (MergeToNaryJoin.mergable(op.left) and
                    MergeToNaryJoin.mergable(op.right))

    @staticmethod
    def collect_join_groups(op, conditions, children):
        assert isinstance(op, algebra.ProjectingJoin)
        assert (isinstance(op.right, algebra.Select)
                or issubclass(type(op.right), algebra.ZeroaryOperator))
        children.append(op.right)
        conjuncs = expression.extract_conjuncs(op.condition)
        for cond in conjuncs:
            conditions.get_or_insert(cond.left)
            conditions.get_or_insert(cond.right)
            conditions.union(cond.left, cond.right)
        scan_then_select = (isinstance(op.left, algebra.Select) and
                            isinstance(op.left.input, algebra.ZeroaryOperator))
        if (scan_then_select or
                issubclass(type(op.left), algebra.ZeroaryOperator)):
            children.append(op.left)
        else:
            assert isinstance(op.left, algebra.ProjectingJoin)
            MergeToNaryJoin.collect_join_groups(op.left, conditions, children)

    def fire(self, op):
        if not isinstance(op, algebra.ProjectingJoin):
            return op

        # if op is the only binary join, return
        if not isinstance(op.left, algebra.ProjectingJoin):
            return op
        # if it is not mergable, e.g. aggregation along the path, return
        if not MergeToNaryJoin.mergable(op):
            return op
        # do the actual merge
        # 1. collect join groups
        join_groups = UnionFind()
        children = []
        MergeToNaryJoin.collect_join_groups(
            op, join_groups, children)
        # 2. extract join groups from the union-find data structure
        join_conds = defaultdict(list)
        for field, key in join_groups.parents.items():
            join_conds[key].append(field)
        conditions = [v for (k, v) in join_conds.items()]
        # Note: a cost based join order optimization need to be implemented.
        ordered_conds = sorted(conditions, key=lambda cond: min(cond))
        # 3. reverse the children due to top-down tree traversal
        return algebra.NaryJoin(
            list(reversed(children)), ordered_conds, op.output_columns)


class GetCardinalities(rules.Rule):
    """ get cardinalities information of Zeroary operators.
    """
    def __init__(self, catalog):
        assert isinstance(catalog, Catalog)
        self.catalog = catalog

    def fire(self, expr):
        # if not Zeroary operator, who cares?
        if not issubclass(type(expr), algebra.ZeroaryOperator):
            return expr

        if issubclass(type(expr), algebra.Scan):
            rel = expr.relation_key
            expr._cardinality = self.catalog.num_tuples(rel)
            return expr
        expr._cardinality = algebra.DEFAULT_CARDINALITY
        return expr


# 6. shuffle logics, hyper_cube_shuffle_logic is only used in HCAlgebra
left_deep_tree_shuffle_logic = [
    ShuffleBeforeSetop(),
    ShuffleBeforeJoin(),
    BroadcastBeforeCross()
]

# 7. distributed groupby
# this need to be put after shuffle logic
distributed_group_by = [
    # DecomposeGroupBy may introduce a complex GroupBy,
    # so we must run SimpleGroupBy after it. TODO no one likes this.
    DecomposeGroupBy(),
    rules.SimpleGroupBy(),
    rules.CountToCountall(),   # TODO revisit when we have NULL support.
    rules.DedupGroupBy(),
    rules.EmptyGroupByToDistinct(),
]

# 8. Myriafy logical operators
# replace logical operator with its corresponding Myria operators
myriafy = [
    rules.OneToOne(algebra.CrossProduct, MyriaCrossProduct),
    rules.OneToOne(algebra.Store, MyriaStore),
    rules.OneToOne(algebra.StoreTemp, MyriaStoreTemp),
    rules.OneToOne(algebra.StatefulApply, MyriaStatefulApply),
    rules.OneToOne(algebra.Apply, MyriaApply),
    rules.OneToOne(algebra.Select, MyriaSelect),
    rules.OneToOne(algebra.Distinct, MyriaDupElim),
    rules.OneToOne(algebra.Shuffle, MyriaShuffle),
    rules.OneToOne(algebra.HyperCubeShuffle, MyriaHyperShuffle),
    rules.OneToOne(algebra.Collect, MyriaCollect),
    rules.OneToOne(algebra.ProjectingJoin, MyriaSymmetricHashJoin),
    rules.OneToOne(algebra.NaryJoin, MyriaLeapFrogJoin),
    rules.OneToOne(algebra.Scan, MyriaScan),
    rules.OneToOne(algebra.ScanTemp, MyriaScanTemp),
    rules.OneToOne(algebra.SingletonRelation, MyriaSingleton),
    rules.OneToOne(algebra.EmptyRelation, MyriaEmptyRelation),
    rules.OneToOne(algebra.UnionAll, MyriaUnionAll),
    rules.OneToOne(algebra.Difference, MyriaDifference),
    rules.OneToOne(algebra.OrderBy, MyriaInMemoryOrderBy),
]

# 9. break communication boundary
# get producer/consumer pair
break_communication = [
    BreakHyperCubeShuffle(),
    BreakShuffle(),
    BreakCollect(),
    BreakBroadcast(),
]


class MyriaAlgebra(Algebra):
    """ Myria algebra abstract class"""
    language = MyriaLanguage

    fragment_leaves = (
        MyriaShuffleConsumer,
        MyriaCollectConsumer,
        MyriaBroadcastConsumer,
        MyriaHyperShuffleConsumer,
        MyriaScan,
        MyriaScanTemp
    )


class MyriaLeftDeepTreeAlgebra(MyriaAlgebra):
    """Myria physical algebra using left deep tree pipeline and 1-D shuffle"""
    def opt_rules(self, **kwargs):
        opt_grps_sequence = [
            rules.remove_trivial_sequences,
            [
                rules.SimpleGroupBy(),
                rules.CountToCountall(),  # TODO revisit when we have NULLs
                rules.ProjectToDistinctColumnSelect(),
                rules.DistinctToGroupBy(),
                rules.DedupGroupBy(),
            ],
            rules.push_select,
            rules.push_project,
            rules.push_apply,
            left_deep_tree_shuffle_logic,
            distributed_group_by,
            [rules.PushApply()],
        ]

        if kwargs.get('push_sql', False):
            opt_grps_sequence.append([
                PushIntoSQL(dialect=kwargs.get('dialect'))])

        compile_grps_sequence = [
            myriafy,
            [AddAppendTemp()],
            break_communication
        ]

        rule_grps_sequence = opt_grps_sequence + compile_grps_sequence
        return list(itertools.chain(*rule_grps_sequence))


class MyriaHyperCubeAlgebra(MyriaAlgebra):
    """Myria physical algebra using HyperCubeShuffle and LeapFrogJoin"""
    def opt_rules(self, **kwargs):
        # this rule is hyper cube shuffle specific
        merge_to_nary_join = [
            MergeToNaryJoin()
        ]

        # catalog aware hc shuffle rules, so put them here
        hyper_cube_shuffle_logic = [
            GetCardinalities(self.catalog),
            HCShuffleBeforeNaryJoin(self.catalog),
            OrderByBeforeNaryJoin(),
        ]

        opt_grps_sequence = [
            rules.remove_trivial_sequences,
            [
                rules.SimpleGroupBy(),
                # TODO revisit when we have NULL support.
                rules.CountToCountall(),
                rules.DistinctToGroupBy(),
                rules.DedupGroupBy(),
            ],
            rules.push_select,
            rules.push_project,
            merge_to_nary_join,
            rules.push_apply,
            left_deep_tree_shuffle_logic,
            distributed_group_by,
            hyper_cube_shuffle_logic
        ]

        if kwargs.get('push_sql', False):
            opt_grps_sequence.append([PushIntoSQL()])

        compile_grps_sequence = [
            myriafy,
            [AddAppendTemp()],
            break_communication
        ]
        rule_grps_sequence = opt_grps_sequence + compile_grps_sequence
        return list(itertools.chain(*rule_grps_sequence))

    def __init__(self, catalog=None):
        self.catalog = catalog


class OpIdFactory(object):
    def __init__(self):
        self.count = 0

    def alloc(self):
        ret = self.count
        self.count += 1
        return ret

    def getter(self):
        return lambda: self.alloc()


def ensure_store_temp(label, op):
    """Returns a StoreTemp that assigns the given operator to a temp relation
    with the given name. Op must be a 'normal operator', i.e.,
    not a Store/StoreTemp or a control-flow sub-plan operator."""
    assert not isinstance(op, (algebra.Store, algebra.StoreTemp))
    assert not isinstance(op, (algebra.Sequence, algebra.Parallel,
                               algebra.DoWhile))
    assert isinstance(label, basestring) and len(label) > 0
    return MyriaStoreTemp(input=op, name=label)


def compile_fragment(frag_root):
    """Given a root operator, produce a SubQueryEncoding."""

    # A dictionary mapping each object to a unique, object-dependent id.
    # Since we want this to be truly unique for each object instance, even if
    # two objects are equal, we use id(obj) as the key.
    opid_factory = OpIdFactory()
    op_ids = defaultdict(opid_factory.getter())

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
        op_id = op_ids[id(op)]
        child_op_ids = [op_ids[id(child)] for child in op.children()]
        op_dict = op.compileme(*child_op_ids)
        op_dict['opName'] = op.shortStr()
        assert isinstance(op_id, int), (type(op_id), op_id)
        op_dict['opId'] = op_id
        return op_dict

    # Determine and encode the fragments.
    return [{'operators': [call_compile_me(op) for op in frag]}
            for frag in fragments(frag_root)]


def compile_plan(plan_op):
    subplan_ops = (algebra.Parallel, algebra.Sequence, algebra.DoWhile)
    if not isinstance(plan_op, subplan_ops):
        plan_op = algebra.Parallel([plan_op])

    if isinstance(plan_op, algebra.Parallel):
        frag_list = [compile_fragment(op) for op in plan_op.children()]
        return {"type": "SubQuery",
                "fragments": list(itertools.chain(*frag_list))}

    elif isinstance(plan_op, algebra.Sequence):
        plan_list = [compile_plan(pl_op) for pl_op in plan_op.children()]
        return {"type": "Sequence", "plans": plan_list}

    elif isinstance(plan_op, algebra.DoWhile):
        children = plan_op.children()
        if len(children) < 2:
            raise ValueError('DoWhile must have at >= 2 children: body and condition')  # noqa
        condition = children[-1]
        if isinstance(condition, subplan_ops):
            raise ValueError('DoWhile condition cannot be a subplan op {cls}'.format(cls=condition.__class__))  # noqa
        condition = ensure_store_temp('__dowhile_{}_condition'.format(id(
            plan_op)), condition)
        plan_op.args = children[:-1] + [condition]
        body = [compile_plan(pl_op) for pl_op in plan_op.children()]
        return {"type": "DoWhile",
                "body": body,
                "condition": condition.name}

    raise NotImplementedError("compiling subplan op {}".format(type(plan_op)))


def compile_to_json(raw_query, logical_plan, physical_plan,
                    language="not specified"):
    """This function compiles a physical query plan to the JSON suitable for
    submission to the Myria REST API server. The logical plan is converted to a
    string and passed along unchanged."""

    # Store/StoreTemp is a reasonable physical plan... for now.
    if isinstance(physical_plan, (algebra.Store, algebra.StoreTemp)):
        physical_plan = algebra.Parallel([physical_plan])

    subplan_ops = (algebra.Parallel, algebra.Sequence, algebra.DoWhile)
    assert isinstance(physical_plan, subplan_ops), \
        'Physical plan must be a subplan operator, not {}'.format(type(physical_plan))  # noqa

    # raw_query must be a string
    if not isinstance(raw_query, basestring):
        raise ValueError("raw query must be a string")

    return {"rawQuery": raw_query,
            "logicalRa": str(logical_plan),
            "language": language,
            "plan": compile_plan(physical_plan)}
