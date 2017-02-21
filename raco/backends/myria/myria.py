import itertools
import logging
from collections import defaultdict, deque
from functools import reduce
from operator import mul

from sqlalchemy.dialects import postgresql

from raco import algebra, expression, rules, scheme
from raco import types
from raco.algebra import Shuffle
from raco.algebra import convertcondition
from raco.backends import Language, Algebra
from raco.backends.sql.catalog import SQLCatalog, PostgresSQLFunctionProvider
from raco.catalog import Catalog
from raco.datastructure.UnionFind import UnionFind
from raco.expression import UnnamedAttributeRef
from raco.expression import WORKERID, COUNTALL
from raco.representation import RepresentationProperties
from raco.rules import distributed_group_by, check_partition_equality
from raco.expression import util

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
        if isinstance(op.value, int):
            myria_type = types.LONG_TYPE
        elif isinstance(op.value, float):
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
    elif isinstance(op, expression.SUBSTR):
        children = []
        op.operands[1] = expression.CAST(types.INT_TYPE, op.operands[1])
        op.operands[2] = expression.CAST(types.INT_TYPE, op.operands[2])
        for operand in op.operands:
            children.append(compile_expr(operand, child_scheme, state_scheme))
        return {
            'type': op.opname(),
            'children': children
        }
    elif isinstance(op, expression.PythonUDF):
        return {
            'type': op.opname(),
            'name': op.name,
            'outputType': op.typ,
            'arguments': [compile_expr(arg, child_scheme, state_scheme)
                          for arg in op.arguments]
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
            "debroadcast": self._debroadcast
        }


class MyriaScanTemp(algebra.ScanTemp, MyriaOperator):

    def compileme(self):
        return {
            "opType": "TempTableScan",
            "table": self.name,
            "debroadcast": self._debroadcast
        }


class MyriaFileScan(algebra.FileScan, MyriaOperator):

    def compileme(self):
        if self.format == 'OPP':
            encoding = dict({
                "opType": "SeaFlowScan",
                "source": self.get_source(self.path)
            }, **self.options)

        elif self.format == 'TIPSY':
            encoding = {
                "opType": "TipsyFileScan",
                "tipsyFilename": self.path,
                "iorderFilename": self.path + ".iord",
                "grpFilename": self.path + (
                    ".grp" if "group" not in self.options
                    else ".{}.grp".format(self.options["group"]))
            }

        else:
            encoding = {
                "opType": "TupleSource",
                "reader": dict({
                    "readerType": "CSV",
                    "schema": scheme_to_schema(self.scheme())
                }, **self.options),
                "source": {
                    "dataType": "URI",
                    "uri": self.path
                }
            }

        return encoding


class MyriaLimit(algebra.Limit, MyriaOperator):

    def compileme(self, inputid):
        return {
            "opType": "Limit",
            "argChild": inputid,
            "numTuples": self.count,
        }


class MyriaUnionAll(algebra.UnionAll, MyriaOperator):

    def compileme(self, *args):
        return {
            "opType": "UnionAll",
            "argChildren": args
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
        distributeFunction = None
        attributes = self.partitioning().hash_partitioned
        if attributes:
            indexes = [attr.get_position(self.scheme()) for attr in attributes]
            distributeFunction = {
                "type": "Hash",
                "indexes": indexes
            }
        elif self.partitioning().broadcasted:
            distributeFunction = {
                "type": "Broadcast"
            }
        else:
            distributeFunction = {
                "type": "RoundRobin"
            }
        return {
            "opType": "DbInsert",
            "relationKey": relation_key_to_json(self.relation_key),
            "argOverwriteTable": True,
            "argChild": inputid,
            "distributeFunction": distributeFunction
        }


class MyriaStoreTemp(algebra.StoreTemp, MyriaOperator):

    def compileme(self, inputid):
        distributeFunction = None
        attributes = self.partitioning().hash_partitioned
        if attributes:
            indexes = [attr.get_position(self.scheme()) for attr in attributes]
            distributeFunction = {
                "type": "Hash",
                "indexes": indexes
            }
        elif self.partitioning().broadcasted:
            distributeFunction = {
                "type": "Broadcast"
            }
        else:
            distributeFunction = {
                "type": "RoundRobin"
            }
        return {
            "opType": "TempInsert",
            "table": self.name,
            "argOverwriteTable": True,
            "argChild": inputid,
            "distributeFunction": distributeFunction
        }


class MyriaSink(algebra.Sink, MyriaOperator):

    def compileme(self, inputid):
        return {
            "opType": "EmptySink",
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
            "argSelect2": allright,
            "argOrder": self.pull_order_policy
        }

        return join


class MyriaIDBController(algebra.IDBController, MyriaOperator):

    def compileme(self, *args):
        ret = {
            "opType": "IDBController",
            "argSelfIdbId": "%s" % self.idb_id,
            "argInitialInput": "%s" % args[0],
            "argIterationInput": "%s" % args[1],
            "argEosControllerInput": "%s" % args[2],
            "argState": self.get_agg(),
            "sync": self.recursion_mode == "SYNC"
        }
        if self.relation_key is not None:
            ret.update({
                "relationKey": relation_key_to_json(self.relation_key),
            })
        return ret


class MyriaEOSController(algebra.EOSController, MyriaOperator):

    def compileme(self, input):
        return {
            "opType": "EOSController",
            "argChild": input
        }


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

        def comp_map(x):
            compile_mapping(x, child_scheme, state_scheme)
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

    def partitioning(self):
        return algebra.Broadcast(self.input).partitioning()

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

    def partitioning(self):
        return self.input.partitioning()

    def shortStr(self):
        return "%s" % self.opname()

    def compileme(self, inputid):
        return {
            'opType': 'BroadcastConsumer',
            'argOperatorId': inputid
        }


class MyriaSplitProducer(algebra.UnaryOperator, MyriaOperator):

    """A Myria SplitProducer"""

    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)
        self.consumers = []

    def append_consumer(self, consumer):
        self.consumers.append(consumer)

    def shortStr(self):
        return self.opname()

    def __repr__(self):
        return "{op}({inp!r})".format(op=self.opname(), inp=self.input)

    def num_tuples(self):
        return self.input.num_tuples()

    def partitioning(self):
        return self.input.partitioning()

    def compileme(self, inputid):
        return {
            "opType": "LocalMultiwayProducer",
            "argChild": inputid
        }


class MyriaSplitConsumer(algebra.UnaryOperator, MyriaOperator):

    """A Myria SplitConsumer"""

    def __init__(self, input):
        assert isinstance(input, MyriaSplitProducer)
        algebra.UnaryOperator.__init__(self, input)
        input.append_consumer(self)

    def num_tuples(self):
        return self.input.num_tuples()

    def partitioning(self):
        return self.input.partitioning()

    def shortStr(self):
        return self.opname()

    def compileme(self, inputid):
        return {
            'opType': 'LocalMultiwayConsumer',
            'argOperatorId': inputid
        }


class MyriaShuffleProducer(algebra.UnaryOperator, MyriaOperator):

    """A Myria ShuffleProducer"""

    def __init__(self, input, hash_columns, shuffle_type=None):
        algebra.UnaryOperator.__init__(self, input)
        # If no specified shuffle type, it's a hash.
        # TODO: add support for more types, do not use None for Hash
        if shuffle_type is None:
            shuffle_type = Shuffle.ShuffleType.Hash
        if shuffle_type == Shuffle.ShuffleType.Identity:
            assert len(hash_columns) == 1
        self.hash_columns = hash_columns
        self.shuffle_type = shuffle_type
        self.buffer_type = None

    def shortStr(self):
        if self.shuffle_type == Shuffle.ShuffleType.Identity:
            return "%s(%s)" % (self.opname(), self.hash_columns[0])
        if self.shuffle_type == Shuffle.ShuffleType.RoundRobin:
            return "%s" % self.opname()
        if self.shuffle_type == Shuffle.ShuffleType.Hash:
            hash_string = ','.join([str(x) for x in self.hash_columns])
            return "%s(h(%s))" % (self.opname(), hash_string)

    def __repr__(self):
        return "{op}({inp!r}, {hc!r}, {st!r})".format(
            op=self.opname(), inp=self.input, hc=self.hash_columns,
            st=self.shuffle_type)

    def partitioning(self):
        return Shuffle(
            self.input,
            self.hash_columns,
            self.shuffle_type).partitioning()

    def set_buffer_type(self, buffer_type):
        self.buffer_type = buffer_type

    def num_tuples(self):
        return self.input.num_tuples()

    def compileme(self, inputid):
        if self.shuffle_type == Shuffle.ShuffleType.Hash:
            df = {
                "type": "Hash",
                "indexes": [x.position for x in self.hash_columns]
            }
        elif self.shuffle_type == Shuffle.ShuffleType.Identity:
            df = {
                "type": "Identity",
                "index": self.hash_columns[0].position
            }
        elif self.shuffle_type == Shuffle.ShuffleType.RoundRobin:
            df = {
                "type": "RoundRobin"
            }
        else:
            # TODO: merge HyperCubeShuffleProducer
            raise ValueError("Invalid ShuffleType")

        ret = {
            "opType": "ShuffleProducer",
            "argChild": inputid,
            "distributeFunction": df
        }
        if self.buffer_type is not None:
            ret.update({"argBufferStateType": self.buffer_type})

        return ret


class MyriaShuffleConsumer(algebra.UnaryOperator, MyriaOperator):

    """A Myria ShuffleConsumer"""

    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)

    def num_tuples(self):
        return self.input.num_tuples()

    def shortStr(self):
        return "%s" % self.opname()

    def partitioning(self):
        return self.input.partitioning()

    def compileme(self, inputid):
        return {
            'opType': 'ShuffleConsumer',
            'argOperatorId': inputid
        }


class MyriaConsumer(algebra.UnaryOperator, MyriaOperator):

    """A Myria Consumer. Used as child for IDBController and EOSController"""

    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)
        self.set_stop_recursion()

    def __repr__(self):
        return "{op}({inp!r})".format(inp=self.input, op=self.opname())

    def partitioning(self):
        return self.input.partitioning()

    def num_tuples(self):
        return self.input.num_tuples()

    def shortStr(self):
        return "%s(%s)" % (self.opname(), self.scheme())

    def scheme(self):
        if isinstance(self.input, MyriaIDBController):
            return scheme.Scheme([("idbID", types.INT_TYPE),
                                  ("isDeltaEmpty", types.BOOLEAN_TYPE)])
        return scheme.Scheme()

    def compileme(self, inputid):
        return {
            'opType': 'Consumer',
            'argOperatorId': inputid
        }


class MyriaCollectProducer(algebra.UnaryOperator, MyriaOperator):

    """A Myria CollectProducer"""

    def __init__(self, input, server):
        algebra.UnaryOperator.__init__(self, input)
        self.server = server

    def num_tuples(self):
        return self.input.num_tuples()

    def partitioning(self):
        # TODO: have a way to say it is on a specific worker
        return RepresentationProperties()

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

    def partitioning(self):
        return self.input.partitioning()

    def shortStr(self):
        return "%s" % self.opname()

    def compileme(self, inputid):
        return {
            'opType': 'CollectConsumer',
            'argOperatorId': inputid
        }


class MyriaHyperCubeShuffle(algebra.HyperCubeShuffle, MyriaOperator):

    """Represents a HyperCubeShuffle shuffle operator"""

    def compileme(self, inputsym):
        raise NotImplementedError('shouldn''t ever get here, should be turned into HCSP-HCSC pair')  # noqa


class MyriaHyperCubeShuffleProducer(algebra.UnaryOperator, MyriaOperator):

    """A Myria HyperCubeShuffleProducer"""

    def __init__(self, input, hashed_columns,
                 hyper_cube_dims, mapped_hc_dims, cell_partition):
        algebra.UnaryOperator.__init__(self, input)
        self.hashed_columns = hashed_columns
        self.mapped_hc_dimensions = mapped_hc_dims
        self.hyper_cube_dimensions = hyper_cube_dims
        self.cell_partition = cell_partition

    def num_tuples(self):
        return self.input.num_tuples()

    def partitioning(self):
        return MyriaHyperCubeShuffle(
            self.input,
            self.hashed_columns,
            self.mapped_hc_dims,
            self.hyper_cube_dimensions,
            self.cell_partition).partitioning()

    def shortStr(self):
        mapping = {i: '*' for i in range(len(self.hyper_cube_dimensions))}
        mapping.update({h: 'h({col})'.format(col=i)
                        for i, h in zip(self.hashed_columns,
                                        self.mapped_hc_dimensions)})
        hash_string = ','.join(s for m, s in sorted(mapping.items()))
        return "%s(%s)" % (self.opname(), hash_string)

    def compileme(self, inputsym):
        return {
            "opType": "HyperCubeShuffleProducer",
            "distributeFunction": {
                "hashedColumns": list(self.hashed_columns),
                "mappedHCDimensions": list(self.mapped_hc_dimensions),
                "hyperCubeDimensions": list(self.hyper_cube_dimensions),
                "cellPartition": self.cell_partition
            },
            "argChild": inputsym
        }


class MyriaHyperCubeShuffleConsumer(algebra.UnaryOperator, MyriaOperator):

    """A Myria HyperCubeShuffleConsumer"""

    def __init__(self, input):
        algebra.UnaryOperator.__init__(self, input)

    def num_tuples(self):
        return self.input.num_tuples()

    def partitioning(self):
        return self.input.partitioning()

    def shortStr(self):
        return "%s" % self.opname()

    def compileme(self, inputsym):
        return {
            "opType": "HyperCubeShuffleConsumer",
            "argOperatorId": inputsym
        }


class MyriaQueryScan(algebra.ZeroaryOperator, MyriaOperator):

    """A Myria Query Scan"""

    def __init__(self, sql, scheme, source_relation_keys,
                 num_tuples=algebra.DEFAULT_CARDINALITY,
                 partitioning=RepresentationProperties(),
                 debroadcast=False):
        algebra.ZeroaryOperator.__init__(self)
        self.sql = str(sql)
        self._scheme = scheme
        self.source_relation_keys = source_relation_keys
        self._num_tuples = num_tuples
        self._partitioning = partitioning
        self._debroadcast = debroadcast

    def __repr__(self):
        return ("{op}({sql!r}, {sch!r}, {rels!r}, {nt!r}, {part!r}, {db!r})"
                .format(op=self.opname(), sql=self.sql, sch=self._scheme,
                        rels=self.source_relation_keys, nt=self._num_tuples,
                        part=self._partitioning, db=self._debroadcast))

    def num_tuples(self):
        return self._num_tuples

    def partitioning(self):
        return self._partitioning

    def shortStr(self):
        return "MyriaQueryScan({sql!r})".format(sql=self.sql)

    def scheme(self):
        return self._scheme

    def compileme(self):
        return {
            "opType": "DbQueryScan",
            "sql": self.sql,
            "schema": scheme_to_schema(self._scheme),
            "sourceRelationKeys": [
                relation_key_to_json(rk) for rk in self.source_relation_keys],
            "debroadcast": self._debroadcast
        }


class MyriaCalculateSamplingDistribution(algebra.UnaryOperator, MyriaOperator):

    """A Myria SamplingDistribution operator"""

    def __init__(self, input, sample_size, is_pct, sample_type):
        algebra.UnaryOperator.__init__(self, input)
        self.sample_size = sample_size
        self.is_pct = is_pct
        self.sample_type = sample_type

    def __repr__(self):
        return "{op}({inp!r}, {size!r}, {is_pct!r}, {type!r})".format(
            op=self.opname(),
            inp=self.input,
            size=self.sample_size,
            is_pct=self.is_pct,
            type=self.sample_type)

    def shortStr(self):
        pct = '%' if self.is_pct else ''
        return "{op}{type}({size}{pct})".format(op=self.opname(),
                                                type=self.sample_type,
                                                size=self.sample_size,
                                                pct=pct)

    def num_tuples(self):
        return self.input.num_tuples()

    def partitioning(self):
        return RepresentationProperties()

    def scheme(self):
        return self.input.scheme() + scheme.Scheme([('SampleSize',
                                                     types.LONG_TYPE), (
                                                    'SampleType',
                                                    self.sample_type)])

    def compileme(self, inputid):
        size_key = "samplePercentage" if self.is_pct else "sampleSize"
        return {
            "opType": "SamplingDistribution",
            "argChild": inputid,
            size_key: self.sample_size,
            "sampleType": self.sample_type
        }


class MyriaSample(algebra.BinaryOperator, MyriaOperator):

    """A Myria Sample operator"""

    def __init__(self, left, right, sample_size, is_pct, sample_type):
        algebra.BinaryOperator.__init__(self, left, right)
        # sample_size, sample_type, is_pct are just used for displaying.
        self.sample_size = sample_size
        self.is_pct = is_pct
        self.sample_type = sample_type

    def __repr__(self):
        return "{op}({l!r}, {r!r}, {size!r}, {is_pct!r}, {type!r})".format(
            op=self.opname(),
            l=self.left,
            r=self.right,
            size=self.sample_size,
            is_pct=self.is_pct,
            type=self.sample_type)

    def shortStr(self):
        pct = '%' if self.is_pct else ''
        return "{op}{type}({size}{pct})".format(op=self.opname(),
                                                type=self.sample_type,
                                                size=self.sample_size,
                                                pct=pct)

    def num_tuples(self):
        return self.sample_size

    def partitioning(self):
        return RepresentationProperties()

    def scheme(self):
        """The right operator is the one sampled from."""
        return self.right.scheme()

    def compileme(self, leftid, rightid):
        return {
            "opType": "Sample",
            "argChild1": leftid,
            "argChild2": rightid,
        }


class LogicalSampleToDistributedSample(rules.Rule):

    """Converts logical SampleScan to the sequence of physical operators."""

    def fire(self, expr):
        if isinstance(expr, algebra.SampleScan):
            samp_size = expr.sample_size
            is_pct = expr.is_pct
            samp_type = expr.sample_type
            # Each worker computes (WorkerID, LocalCount).
            scan_r = MyriaScan(expr.relation_key, expr.scheme())
            cnt_all = MyriaGroupBy(input=scan_r, aggregate_list=[COUNTALL()])
            apply_wid = MyriaApply([('WorkerID', WORKERID()),
                                    ('WorkerCount', UnnamedAttributeRef(0))],
                                   cnt_all)
            # Master collects the counts and generates a distribution.
            collect = MyriaCollect(apply_wid)
            samp_dist = MyriaCalculateSamplingDistribution(collect, samp_size,
                                                           is_pct, samp_type)
            # Master sends out how much each worker should sample.
            shuff = MyriaShuffle(samp_dist, [UnnamedAttributeRef(0)],
                                 Shuffle.ShuffleType.Identity)
            # Workers perform actual sampling.
            samp = MyriaSample(shuff, scan_r, samp_size, is_pct, samp_type)
            return samp
        else:
            return expr


class BreakShuffle(rules.Rule):

    def fire(self, expr):
        if not isinstance(expr, MyriaShuffle):
            return expr

        producer = MyriaShuffleProducer(expr.input, expr.columnlist,
                                        expr.shuffle_type)
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
        if not isinstance(expr, MyriaHyperCubeShuffle):
            return expr
        producer = MyriaHyperCubeShuffleProducer(
            expr.input, expr.hashed_columns, expr.hyper_cube_dimensions,
            expr.mapped_hc_dimensions, expr.cell_partition)
        consumer = MyriaHyperCubeShuffleConsumer(producer)
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


class BreakSplit(rules.Rule):

    def fire(self, expr):
        if not isinstance(expr, algebra.Split):
            return expr

        producer = MyriaSplitProducer(expr.input)
        consumer = MyriaSplitConsumer(producer)
        return consumer


class CollectBeforeLimit(rules.Rule):

    """Similar to a decomposable GroupBy, rewrite Limit as
    Limit[Collect[Limit]]"""

    def fire(self, exp):
        if exp.__class__ == algebra.Limit:
            return MyriaLimit(exp.count,
                              algebra.Collect(
                                  MyriaLimit(exp.count, exp.input)))

        return exp


class ShuffleBeforeSetop(rules.Rule):

    def fire(self, exp):
        if not isinstance(exp, (algebra.Difference, algebra.Intersection)):
            return exp

        def shuffle_after(op):
            cols = [expression.UnnamedAttributeRef(i)
                    for i in range(len(op.scheme()))]

            if check_partition_equality(op, cols):
                return op
            else:
                return algebra.Shuffle(child=op, columnlist=cols)

        exp.left = shuffle_after(exp.left)
        exp.right = shuffle_after(exp.right)

        return exp

    def __str__(self):
        return "Setop => Shuffle(Setop)"


class ShuffleBeforeJoin(rules.Rule):

    def fire(self, expr):
        # If not a join, who cares?
        if not isinstance(expr, algebra.Join):
            return expr

        # Figure out which columns go in the shuffle
        left_cols, right_cols = \
            convertcondition(expr.condition,
                             len(expr.left.scheme()),
                             expr.left.scheme() + expr.right.scheme())

        # Left shuffle cols
        left_cols = [expression.UnnamedAttributeRef(i)
                     for i in left_cols]
        # Right shuffle cols
        right_cols = [expression.UnnamedAttributeRef(i)
                      for i in right_cols]

        if check_partition_equality(expr.left, left_cols):
            new_left = expr.left
        elif expr.left.partitioning().broadcasted:
            new_left = expr.left
        else:
            new_left = algebra.Shuffle(expr.left, left_cols)

        if check_partition_equality(expr.right, right_cols):
            new_right = expr.right
        elif expr.right.partitioning().broadcasted:
            new_right = expr.right
        else:
            new_right = algebra.Shuffle(expr.right, right_cols)

        # Construct the object!
        assert isinstance(expr, algebra.ProjectingJoin)
        if isinstance(expr, algebra.ProjectingJoin):
            return algebra.ProjectingJoin(expr.condition,
                                          new_left, new_right,
                                          expr.output_columns)

    def __str__(self):
        return "Join => Shuffle(Join)"


class ShuffleBeforeIDBController(rules.Rule):
    def fire(self, expr):
        if not isinstance(expr, algebra.IDBController):
            return expr
        group_list, agg = expr.get_group_agg()

        for idx in range(2):
            if not isinstance(expr.children()[idx],
                              (algebra.Shuffle, algebra.EmptyRelation)):
                expr.args[idx] = algebra.Shuffle(
                    expr.children()[idx],
                    [expression.UnnamedAttributeRef(i) for i in group_list])
        return expr


class HCShuffleBeforeNaryJoin(rules.Rule):

    def __init__(self, catalog):
        assert isinstance(catalog, Catalog)
        self.catalog = catalog
        super(HCShuffleBeforeNaryJoin, self).__init__()

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
        opt_dim_sizes = [0] * len(conditions)
        while len(toVisit) > 0:
            dim_sizes = toVisit.pop()
            workload = this.workload(dim_sizes, child_sizes, r_index)
            if ((workload < min_work_load) or (
                workload == min_work_load and max(dim_sizes) < max(
                    opt_dim_sizes)) or (min_work_load is None)):
                min_work_load = this.workload(
                    dim_sizes, child_sizes, r_index)
                opt_dim_sizes = dim_sizes
            visited.add(dim_sizes)
            for i, d in enumerate(dim_sizes):
                new_dim_sizes = (dim_sizes[0:i] +
                                 tuple([dim_sizes[i] + 1]) +
                                 dim_sizes[i + 1:])
                if (product(new_dim_sizes) <= num_server and
                        new_dim_sizes not in visited):
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
        except NotImplementedError:
            # If cardinalities unknown, broadcast the right child
            expr.right = algebra.Broadcast(expr.right)

        return expr


class ShuffleAfterSingleton(rules.Rule):

    def fire(self, expr):
        if isinstance(expr, MyriaSingleton):
            return expr

        if isinstance(expr, algebra.SingletonRelation):
            return algebra.Shuffle(MyriaSingleton(), [UnnamedAttributeRef(0)])

        return expr


class ShuffleAfterFileScan(rules.Rule):

    def fire(self, expr):
        # don't shuffle any FileScan under an existing shuffle or broadcast
        if isinstance(expr, (algebra.Shuffle,
                             algebra.HyperCubeShuffle,
                             algebra.Broadcast)):
            for e in expr.walk():
                if isinstance(e, algebra.FileScan):
                    e._needs_shuffle = False
            return expr

        if isinstance(expr, algebra.FileScan):
            if expr._needs_shuffle:
                expr._needs_shuffle = False
                return algebra.Shuffle(
                    expr, shuffle_type=Shuffle.ShuffleType.RoundRobin)

        return expr

    def __str__(self):
        return "FileScan => Shuffle(FileScan)"


class PushSelectThroughShuffle(rules.Rule):

    """Push selections through a shuffle. We only need to push it just under
    the shuffle, since we can invoke the PushSelects rule again. We assume that
    PushSelects has already been invoked, so all selects that could be pushed
    through a shuffle are already immediately above the shuffle."""

    def fire(self, expr):
        if (isinstance(expr, algebra.Select) and
            isinstance(expr.input, (algebra.Shuffle,
                                    algebra.HyperCubeShuffle,
                                    algebra.Broadcast,
                                    algebra.Collect))):

            old_select = expr
            old_shuffle = expr.input
            shuffle_child = expr.input.input

            new_select = old_select.__class__()
            new_select.copy(old_select)
            new_select.input = shuffle_child

            new_shuffle = old_shuffle.__class__()
            new_shuffle.copy(old_shuffle)
            new_shuffle.input = new_select

            return new_shuffle

        return expr

    def __str__(self):
        return "Select(Shuffle) => Shuffle(Select)"


class AddAppendTemp(rules.Rule):

    def fire(self, op):
        if not isinstance(op, MyriaStoreTemp):
            return op

        child = op.input
        if not isinstance(child, MyriaUnionAll):
            return op

        # TODO: handle multiple children.
        if len(child.args) != 2:
            return op

        left = child.args[0]
        right = child.args[1]
        rel_name = op.name

        def is_scan(op):
            return (isinstance(op, MyriaScanTemp) and op.name == rel_name)
        if is_scan(left) and not any(is_scan(op) for op in right.walk()):
            return MyriaAppendTemp(name=rel_name, input=right)
        elif is_scan(right) and not any(is_scan(op) for op in left.walk()):
            return MyriaAppendTemp(name=rel_name, input=left)

        return op


class PushIntoSQL(rules.Rule):

    def __init__(self, dialect=None, push_grouping=False):
        self.dialect = dialect or postgresql.dialect()
        self.push_grouping = push_grouping
        super(PushIntoSQL, self).__init__()

    def fire(self, expr):
        if isinstance(expr, (algebra.Scan, algebra.ScanTemp)):
            return expr
        cat = SQLCatalog(provider=PostgresSQLFunctionProvider(),
                         push_grouping=self.push_grouping)
        try:
            scan_relations = [s.relation_key for s in expr.walk()
                              if isinstance(s, algebra.Scan)]
            has_debroadcast = any(isinstance(s, algebra.Scan) and
                                  s._debroadcast for s in expr.walk())
            sql_plan = cat.get_sql(expr)
            sql_string = sql_plan.compile(dialect=self.dialect)
            sql_string.visit_bindparam = sql_string.render_literal_bindparam
            return MyriaQueryScan(sql=sql_string.process(sql_plan),
                                  scheme=expr.scheme(),
                                  source_relation_keys=scan_relations,
                                  num_tuples=expr.num_tuples(),
                                  partitioning=expr.partitioning(),
                                  debroadcast=has_debroadcast)
        except NotImplementedError as e:
            LOGGER.warn("Error converting {plan}: {e}"
                        .format(plan=expr, e=e))
            return expr


class InsertSplit(rules.Rule):

    """Inserts an algebra.Split operator in every fragment that has multiple
    heavy-weight operators."""
    heavy_ops = (algebra.Store, algebra.StoreTemp,
                 algebra.CrossProduct, algebra.Join, algebra.NaryJoin,
                 algebra.GroupBy, algebra.OrderBy)

    def insert_split_before_heavy(self, op):
        """Walk the tree starting from op and insert a split when we
        encounter a heavyweight operator."""
        if isinstance(op, MyriaAlgebra.fragment_leaves):
            return op

        if isinstance(op, InsertSplit.heavy_ops):
            return algebra.Split(op)

        return op.apply(self.insert_split_before_heavy)

    def fire(self, op):
        if isinstance(op, InsertSplit.heavy_ops):
            return op.apply(self.insert_split_before_heavy)

        return op


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
        assert (isinstance(op.right, algebra.Select) or
                issubclass(type(op.right), algebra.ZeroaryOperator))
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
        super(GetCardinalities, self).__init__()

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
    ShuffleBeforeIDBController(),
    BroadcastBeforeCross(),
    ShuffleAfterSingleton(),
    CollectBeforeLimit(),
    ShuffleAfterFileScan(),
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
    rules.OneToOne(algebra.HyperCubeShuffle, MyriaHyperCubeShuffle),
    rules.OneToOne(algebra.Collect, MyriaCollect),
    rules.OneToOne(algebra.ProjectingJoin, MyriaSymmetricHashJoin),
    rules.OneToOne(algebra.NaryJoin, MyriaLeapFrogJoin),
    rules.OneToOne(algebra.Scan, MyriaScan),
    rules.OneToOne(algebra.ScanTemp, MyriaScanTemp),
    rules.OneToOne(algebra.FileScan, MyriaFileScan),
    rules.OneToOne(algebra.SingletonRelation, MyriaSingleton),
    rules.OneToOne(algebra.EmptyRelation, MyriaEmptyRelation),
    rules.OneToOne(algebra.UnionAll, MyriaUnionAll),
    rules.OneToOne(algebra.Difference, MyriaDifference),
    rules.OneToOne(algebra.OrderBy, MyriaInMemoryOrderBy),
    rules.OneToOne(algebra.Sink, MyriaSink),
    rules.OneToOne(algebra.IDBController, MyriaIDBController),
]

# 9. break communication boundary
# get producer/consumer pair
break_communication = [
    BreakHyperCubeShuffle(),
    BreakShuffle(),
    BreakCollect(),
    BreakBroadcast(),
    BreakSplit(),
]


class MyriaAlgebra(Algebra):

    """ Myria algebra abstract class"""
    language = MyriaLanguage

    fragment_leaves = (
        MyriaSplitConsumer,
        MyriaShuffleConsumer,
        MyriaCollectConsumer,
        MyriaBroadcastConsumer,
        MyriaHyperCubeShuffleConsumer,
        MyriaScan,
        MyriaScanTemp,
        MyriaFileScan,
        MyriaEmptyRelation,
        MyriaSingleton
    )


class FlattenUnionAll(rules.Rule):

    @staticmethod
    def collect_children(op):
        if isinstance(op, algebra.UnionAll):
            children = []
            for child in op.args:
                children += FlattenUnionAll.collect_children(child)
            return children
        return [op]

    def fire(self, op):
        if not isinstance(op, algebra.UnionAll):
            return op
        children = FlattenUnionAll.collect_children(op)
        if len(children) == 1:
            return children[0]
        return algebra.UnionAll(children)


class FillInJoinPullOrder(rules.Rule):

    def __init__(self):
        self._disabled = False

    def find_idb(self, op):
        for child in op.children():
            if isinstance(child, MyriaIDBController):
                return True
            if self.find_idb(child):
                return True
        return False

    def fire(self, op):
        if not isinstance(op, algebra.UntilConvergence):
            return op
        if op.pull_order_policy == 'ALTERNATE':
            return op
        for ch in op.walk():
            if isinstance(ch, MyriaSymmetricHashJoin):
                left_idb = self.find_idb(ch.left)
                right_idb = self.find_idb(ch.right)
                if op.pull_order_policy == 'PULL_IDB':
                    # ALTERNATE when both are IDBs
                    if left_idb and not right_idb:
                        ch.pull_order_policy = 'LEFT'
                    elif right_idb and not left_idb:
                        ch.pull_order_policy = 'RIGHT'
                elif op.pull_order_policy == 'PULL_EDB':
                    # ALTERNATE when both are EDBs
                    if left_idb and not right_idb:
                        ch.pull_order_policy = 'RIGHT'
                    elif right_idb and not left_idb:
                        ch.pull_order_policy = 'LEFT'
                elif op.pull_order_policy == 'BUILD_EDB':
                    # pick any EDB to save one hash table
                    if not left_idb:
                        ch.pull_order_policy = 'LEFT_EOS'
                    elif not right_idb:
                        ch.pull_order_policy = 'RIGHT_EOS'
        return op


class PropagateAsyncFTBuffer(rules.Rule):

    def __init__(self):
        self._disabled = False

    def fire(self, op):
        if not isinstance(op, MyriaIDBController):
            return op
        buffer_type = op.get_agg()
        if buffer_type['type'] == "CountFilter":
            # do not propagate CountFilter
            return op
        # only do it for the iterative input
        if isinstance(op.args[1], MyriaShuffleConsumer):
            op.args[1].input.set_buffer_type(buffer_type)
        return op


class StoreFromIDB(rules.Rule):

    @staticmethod
    def collect_and_replace(op, idbproducers):
        if not isinstance(op, algebra.NaryOperator):
            return
        if isinstance(op, algebra.UntilConvergence):
            for idbproducer in op.children():
                if isinstance(idbproducer.input, MyriaIDBController):
                    idbproducers[idbproducer.input.name] = idbproducer
            return
        newchildren = []
        for child in op.children():
            if isinstance(child, algebra.UntilConvergence):
                StoreFromIDB.collect_and_replace(child, idbproducers)
            if (isinstance(child, algebra.Store) and
                    isinstance(child.input, algebra.ScanIDB)):
                assert child.input.name in idbproducers
                idbproducers[child.input.name].input.relation_key = \
                    child.relation_key
            else:
                newchildren.append(child)
        op.args = newchildren

    def fire(self, op):
        self.collect_and_replace(op, {})
        return op

    def __str__(self):
        return ("Store[relation](IDBScan(IDBController)) -> "
                "IDBController[relation]")


def replace_child_with(op, child, replacement):
    if isinstance(op, algebra.UnaryOperator):
        op.input = replacement
    elif isinstance(op, algebra.BinaryOperator):
        if op.left == child:
            op.left = replacement
        else:
            op.right = replacement
    elif isinstance(op, algebra.NaryOperator):
        op.args[op.args.index(child)] = replacement
    else:  # should not happen
        assert False


class RemoveEmptyFilter(rules.Rule):

    def fire(self, op):
        for child in op.children():
            if isinstance(child, algebra.Select) and child.condition is None:
                replace_child_with(op, child, child.children()[0])
        return op

    def __str__(self):
        return "Select[cond=null](Input) -> Input"


class RemoveSingleSplit(rules.Rule):

    def fire(self, op):
        if not isinstance(op, algebra.UntilConvergence):
            return op

        parent_map = {}
        op.collectParents(parent_map)
        for idx, child in enumerate(op.children()):
            if not isinstance(child, MyriaSplitProducer):
                continue
            if len(child.consumers) != 1:
                continue
            producer = parent_map[id(child.consumers[0])][0]
            if not isinstance(producer, MyriaShuffleProducer):
                continue
            idb = child.children()[0]
            replace_child_with(producer, child.consumers[0], idb)
            if child.consumers[0].stop_recursion:
                parent_map[id(producer)][0].set_stop_recursion()
            op.args[idx] = producer
        return op

    def __str__(self):
        return ("ShuffleConsumer(ShuffleProducer(SplitConsumer(SplitProducer"
                "(input)))) -> ShuffleConsumer(ShuffleProducer(input))")


class DoUntilConvergence(rules.Rule):

    @staticmethod
    def replace_scan_with_idb_consumer(op, idb_controllers, idb_producers):
        for ch in op.children():
            if isinstance(ch, algebra.ScanIDB):
                if ch.name not in idb_producers:
                    idb_producers[ch.name] = MyriaSplitProducer(
                        idb_controllers[ch.name])
                consumer = MyriaSplitConsumer(idb_producers[ch.name])
                consumer.set_stop_recursion()
                replace_child_with(op, ch, consumer)
                if (isinstance(op, algebra.Select) and
                        (isinstance(op.condition, expression.boolean.GTEQ) or
                         isinstance(op.condition, expression.boolean.GT))):
                    group_list, agg = idb_controllers[ch.name].get_group_agg()
                    if isinstance(agg, expression.aggregate.COUNTALL):
                        if isinstance(op.condition, expression.boolean.GTEQ):
                            agg.threshold = op.condition.right.value
                        else:
                            # count generates integers, translate it to GTEQ
                            agg.threshold = op.condition.right.value + 1
                        op.condition = None
            elif ch.stop_recursion:
                pass
            else:
                DoUntilConvergence.replace_scan_with_idb_consumer(
                    ch, idb_controllers, idb_producers)

    def fire(self, op):
        if not isinstance(op, algebra.UntilConvergence):
            return op
        if any(isinstance(ch, MyriaEOSController) for ch in op.children()):
            return op

        eos_controller = MyriaEOSController()
        eos_consumers = []
        idb_controllers = {}
        for idb_controller in op.children():
            idb_controller.args[2] = MyriaConsumer(eos_controller)
            eos_consumers.append(MyriaConsumer(idb_controller))
            idb_controllers[idb_controller.name] = idb_controller
        if len(eos_consumers) == 1:
            eos_controller.input = eos_consumers[0]
        else:
            eos_controller.input = MyriaUnionAll(eos_consumers)

        idb_producers = {}
        for idb_controller in op.children():
            DoUntilConvergence.replace_scan_with_idb_consumer(
                idb_controller, idb_controllers, idb_producers)
        new_statements = [eos_controller]
        for idb_controller in op.children():
            if idb_controller.name in idb_producers:
                producer = idb_producers[idb_controller.name]
                producer.idx = len(new_statements)
                producer.parent = op
                new_statements.append(producer)
            else:
                new_statements.append(MyriaSink(idb_controller))
        op.args = new_statements
        return op

    def __str__(self):
        return ("DoUntilConvergence")


def idb_until_convergence(async_ft):
    ret = [DoUntilConvergence(), RemoveEmptyFilter(), StoreFromIDB(),
           RemoveSingleSplit(), FillInJoinPullOrder()]
    if async_ft is not None:
        ret.append(PropagateAsyncFTBuffer())
    return ret


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
            [PushSelectThroughShuffle()],
            rules.push_select,
            distributed_group_by(MyriaGroupBy),
            [rules.PushApply()],
            [LogicalSampleToDistributedSample()],
            [FlattenUnionAll()],
            [rules.DeDupBroadcastInputs()],
        ]

        if kwargs.get('push_sql', False):
            opt_grps_sequence.append([
                PushIntoSQL(dialect=kwargs.get('dialect'),
                            push_grouping=kwargs.get(
                                'push_sql_grouping', False))])

        compile_grps_sequence = [
            myriafy,
            [AddAppendTemp()],
            break_communication,
            idb_until_convergence(kwargs.get('async_ft')),
        ]

        if kwargs.get('add_splits', True):
            compile_grps_sequence.append([InsertSplit()])
        # Even when false, plans may already include (manually added) Splits,
        # so we always need BreakSplit
        compile_grps_sequence.append([BreakSplit()])

        rule_grps_sequence = opt_grps_sequence + compile_grps_sequence

        # flatten the rules lists
        rule_list = list(itertools.chain(*rule_grps_sequence))

        # disable specified rules
        rules.Rule.apply_disable_flags(rule_list, *kwargs.keys())

        return rule_list


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
            [PushSelectThroughShuffle()],
            rules.push_select,
            distributed_group_by(MyriaGroupBy),
            [rules.DeDupBroadcastInputs()],
            hyper_cube_shuffle_logic
        ]

        if kwargs.get('push_sql', False):
            opt_grps_sequence.append([PushIntoSQL()])

        compile_grps_sequence = [
            myriafy,
            [AddAppendTemp()],
            break_communication
        ]

        if kwargs.get('add_splits', True):
            compile_grps_sequence.append([InsertSplit()])
        # Even when false, plans may already include (manually added) Splits,
        # so we always need BreakSplit
        compile_grps_sequence.append([BreakSplit()])

        rule_grps_sequence = opt_grps_sequence + compile_grps_sequence

        # flatten the rules lists
        rule_list = list(itertools.chain(*rule_grps_sequence))

        # disable specified rules
        rules.Rule.apply_disable_flags(rule_list, *kwargs.keys())

        return rule_list

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
    op_ids = defaultdict(OpIdFactory().getter())

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
        if rootOp.stop_recursion:
            pass
        elif isinstance(rootOp, MyriaAlgebra.fragment_leaves):
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
            ret.append(list(reversed(op_frag)))
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
    results = []
    for frag in fragments(frag_root):
        frag_compilated = {'operators': [call_compile_me(op) for op in frag]}
        results.append(frag_compilated)

    return results


def compile_plan(plan_op):
    """Given a root operator in MyriaX physical algebra,
    produce the dictionary encoding of the physical plan, in other words, a
    nested collection of Java QueryPlan operators."""

    subplan_ops = (algebra.Parallel, algebra.Sequence, algebra.DoWhile,
                   algebra.UntilConvergence)
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

    elif isinstance(plan_op, algebra.UntilConvergence):
        frag_list = [compile_fragment(op) for op in plan_op.children()]
        return {"type": "SubQuery",
                "fragments": list(itertools.chain(*frag_list))}

    raise NotImplementedError("compiling subplan op {}".format(type(plan_op)))


def compile_to_json(raw_query, logical_plan, physical_plan,
                    language="not specified", **kwargs):
    """This function compiles a physical query plan to the JSON suitable for
    submission to the Myria REST API server. The logical plan is converted to a
    string and passed along unchanged."""

    # Store/StoreTemp is a reasonable physical plan... for now.
    root_ops = (algebra.Store, algebra.StoreTemp, algebra.Sink)
    if isinstance(physical_plan, root_ops):
        physical_plan = algebra.Parallel([physical_plan])

    subplan_ops = (algebra.Parallel, algebra.Sequence, algebra.DoWhile,
                   algebra.UntilConvergence)
    assert isinstance(physical_plan, subplan_ops), \
        'Physical plan must be a subplan operator, not {}'.format(type(physical_plan))  # noqa

    # raw_query must be a string
    if not isinstance(raw_query, basestring):
        raise ValueError("raw query must be a string")

    return {"rawQuery": raw_query, "logicalRa": str(logical_plan),
            "language": language, "plan": compile_plan(physical_plan),
            "ftMode": kwargs.get('async_ft', "NONE")}
