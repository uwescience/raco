import copy
import itertools
from collections import defaultdict, deque
from operator import mul
from abc import abstractmethod

from raco import algebra, expression, rules
from raco.catalog import Catalog
from raco.language import Language
from raco.utility import emit
from raco.expression import (accessed_columns, to_unnamed_recursive,
                             UnnamedAttributeRef)
from raco.expression.aggregate import DecomposableAggregate
from raco.datastructure.UnionFind import UnionFind
from raco import types


def scheme_to_schema(s):
    if s:
        names, descrs = zip(*s.asdict.items())
        names = ["%s" % n for n in names]
        types = [r[1] for r in descrs]
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
            if (2 ** 31) - 1 >= op.value >= -2 ** 31:
                myria_type = 'INT_TYPE'
            else:
                myria_type = types.LONG_TYPE
        elif type(op.value) == float:
            myria_type = types.DOUBLE_TYPE
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

    def compileme(self, inputid):
        child_scheme = self.input.scheme()
        group_fields = [expression.toUnnamed(ref, child_scheme)
                        for ref in self.grouping_list]

        agg_fields = []
        for expr in self.aggregate_list:
            if isinstance(expr, expression.COUNTALL):
                # XXX Wrong in the presence of nulls
                agg_fields.append(UnnamedAttributeRef(0))
            else:
                agg_fields.append(expression.toUnnamed(
                    expr.input, child_scheme))

        agg_types = [[MyriaGroupBy.agg_mapping(agg_expr)]
                     for agg_expr in self.aggregate_list]
        ret = {
            "argChild": inputid,
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


class ShuffleBeforeDistinct(rules.Rule):
    def fire(self, exp):
        if not isinstance(exp, algebra.Distinct):
            return exp
        if isinstance(exp.input, algebra.Shuffle):
            return exp
        cols = [expression.UnnamedAttributeRef(i)
                for i in range(len(exp.scheme()))]
        exp.input = algebra.Shuffle(child=exp.input, columnlist=cols)
        return exp


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

        # By default, broadcast the right child
        expr.right = algebra.Broadcast(expr.right)

        return expr


class DistributedGroupBy(rules.Rule):
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

        local_aggs = []  # aggregates executed on each local machine
        merge_aggs = []  # aggregates executed after local aggs
        agg_offsets = defaultdict(list)  # map aggregate to local agg indices

        for (i, logical_agg) in enumerate(op.aggregate_list):
            for local, merge in zip(logical_agg.get_local_aggregates(),
                                    logical_agg.get_merge_aggregates()):
                try:
                    idx = local_aggs.index(local)
                    agg_offsets[i].append(idx)
                except ValueError:
                    agg_offsets[i].append(len(local_aggs))
                    local_aggs.append(local)
                    merge_aggs.append(merge)

        assert len(merge_aggs) == len(local_aggs)

        local_gb = MyriaGroupBy(op.grouping_list, local_aggs, op.input)

        # Create a merge aggregate; grouping terms are passed through.
        merge_groupings = [UnnamedAttributeRef(i)
                           for i in range(num_grouping_terms)]

        # Connect the output of local aggregates to merge aggregates
        for pos, agg in enumerate(merge_aggs, num_grouping_terms):
            agg.input = UnnamedAttributeRef(pos)

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
            offsets = [idx + num_grouping_terms for idx in agg_offsets[pos]]

            if fexpr is None:
                assert len(offsets) == 1
                return UnnamedAttributeRef(offsets[0])
            else:
                # Convert MergeAggregateOutput instances to absolute col refs
                return expression.finalizer_expr_to_absolute(fexpr, offsets)

        # pass through grouping terms
        gmappings = [(None, UnnamedAttributeRef(i))
                     for i in range(len(op.grouping_list))]
        # extract a single result for aggregate terms
        fmappings = [(None, resolve_finalizer_expr(agg, pos)) for pos, agg in
                     enumerate(op.aggregate_list)]
        return algebra.Apply(gmappings + fmappings, op_out)


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
        # 2. extract join groups from the union find datastructure
        join_conds = defaultdict(list)
        for field, key in join_groups.parents.items():
            join_conds[key].append(field)
        conditions = [sorted(v) for (k, v) in join_conds.items()]
        # Note: a cost based join order optimization need to be implemented.
        ordered_conds = sorted(conditions, key=lambda cond: cond[0])
        # 3. reverse the children due to top-down tree traversal
        naryJoin = algebra.NaryJoin(
            list(reversed(children)), ordered_conds, op.output_columns)
        naryJoin.left_deep_tree_join = op
        return naryJoin


class NaryJoinToLeftDeepTree(rules.Rule):
    """replace NaryJoin with left deep tree of binary joins"""
    def fire(self, op):
        # if op is not NaryJoin, who cares?
        if not isinstance(op, algebra.NaryJoin):
            return op
        # recover binary joins locally
        newop = op.left_deep_tree_join

        # replace the input relations with HyperCubeShuffle
        def replace_child(join, children):
            assert children
            assert isinstance(newop, algebra.ProjectingJoin)
            child = children.pop()
            join.right = child
            if len(children) == 1:
                join.left = children[0]
            else:
                replace_child(join.left, children)

        # replace from right to left
        new_children = list(op.children())
        replace_child(newop, new_children)
        return newop


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

# logical groups of catalog transparent rules
# 1. this must be applied first
remove_trivial_sequences = [rules.RemoveTrivialSequences()]

# 2. simple group by
simple_group_by = [rules.SimpleGroupBy()]

# 3. push down selection
push_select = [
    rules.SplitSelects(),
    rules.PushSelects(),
    rules.MergeSelects()
]

# 4. push projection
push_project = [
    rules.ProjectingJoin(),
    rules.JoinToProjectingJoin()
]

# 5. push apply
push_apply = [
    # These really ought to be run until convergence.
    # For now, run twice and finish with PushApply.
    rules.PushApply(),
    rules.RemoveUnusedColumns(),
    rules.PushApply(),
    rules.RemoveUnusedColumns(),
    rules.PushApply(),
]

# 6. shuffle logics, hyper_cube_shuffle_logic is only used in HCAlgebra
left_deep_tree_shuffle_logic = [
    ShuffleBeforeDistinct(),
    ShuffleBeforeSetop(),
    ShuffleBeforeJoin(),
    BroadcastBeforeCross()
]

# 7. distributed groupby
# this need to be put after shuffle logic
distributed_group_by = [
    # DistributedGroupBy may introduce a complex GroupBy,
    # so we must run SimpleGroupBy after it. TODO no one likes this.
    DistributedGroupBy(), rules.SimpleGroupBy(),
    ProjectToDistinctColumnSelect()
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


class MyriaAlgebra(object):
    """ Myria algebra abstract class
    """
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
        MyriaHyperShuffleConsumer,
        MyriaScan,
        MyriaScanTemp
    )

    @abstractmethod
    def opt_rules(self):
        """Specific Myria algebra must instantiate this method."""


class MyriaLeftDeepTreeAlgebra(MyriaAlgebra):
    """Myria physical algebra using left deep tree pipeline and 1-D shuffle"""
    rule_grps_sequence = [
        remove_trivial_sequences,
        simple_group_by,
        push_select,
        push_project,
        push_apply,
        left_deep_tree_shuffle_logic,
        distributed_group_by,
        myriafy,
        break_communication
    ]

    def opt_rules(self):
        return list(itertools.chain(*self.rule_grps_sequence))


class MyriaHyperCubeAlgebra(MyriaAlgebra):
    """Myria physical algebra using HyperCubeShuffle and LeapFrogJoin"""
    def opt_rules(self):
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

        rule_grps_sequence = [
            remove_trivial_sequences,
            simple_group_by,
            push_select,
            push_project,
            merge_to_nary_join,
            push_apply,
            left_deep_tree_shuffle_logic,
            distributed_group_by,
            hyper_cube_shuffle_logic,
            myriafy,
            break_communication
        ]
        return list(itertools.chain(*rule_grps_sequence))

    def __init__(self, catalog=None):
        self.catalog = catalog


class MyriaHyperCubeLeftDeepTreeJoinAlgebra(MyriaAlgebra):
    """Myria physical algebra using HyperCube shuffle and pipelined joins"""
    def opt_rules(self):
        merge_to_nary_join = [
            MergeToNaryJoin()
        ]

        hyper_cube_shuffle_logic = [
            GetCardinalities(self.catalog),
            HCShuffleBeforeNaryJoin(self.catalog)
        ]

        left_deep_tree_locally = [
            NaryJoinToLeftDeepTree()
        ]

        rule_grps_sequence = [
            remove_trivial_sequences,
            simple_group_by,
            push_select,
            push_project,
            merge_to_nary_join,
            push_apply,
            left_deep_tree_shuffle_logic,
            distributed_group_by,
            hyper_cube_shuffle_logic,
            left_deep_tree_locally,
            myriafy,
            break_communication
        ]
        return list(itertools.chain(*rule_grps_sequence))

    def __init__(self, catalog=None):
        self.catalog = catalog


class MyriaRegularShuffleLeapFrogAlgebra(MyriaAlgebra):
    """Myria phyiscal algebra with regular shuffle and LeapFrogJoin"""
    rules_grps_sequence = [
        remove_trivial_sequences,
        simple_group_by,
        push_select,
        push_project,
        push_apply,
        left_deep_tree_shuffle_logic,
        distributed_group_by,
        myriafy,
        break_communication
    ]


class MyriaBroadcastLeftDeepTreeJoinAlgebra(MyriaAlgebra):
    """Myria phyiscal algebra with broadcast and left deep tree join """
    rules_grps_sequence = [
        remove_trivial_sequences,
        simple_group_by,
        push_select,
        push_project,
        push_apply,
        left_deep_tree_shuffle_logic,
        distributed_group_by,
        myriafy,
        break_communication
    ]


class MyriaBroadCastLeapFrogJoinAlgebra(MyriaAlgebra):
    """Myria phyiscal algebra with broadcast and left deep tree join """
    rules_grps_sequence = [
        remove_trivial_sequences,
        simple_group_by,
        push_select,
        push_project,
        push_apply,
        left_deep_tree_shuffle_logic,
        distributed_group_by,
        myriafy,
        break_communication
    ]


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
