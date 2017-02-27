import raco.myrial.groupby as groupby
import raco.myrial.multiway as multiway
from raco.myrial.cfg import ControlFlowGraph
from raco.myrial.emitarg import FullWildcardEmitArg, TableWildcardEmitArg
from raco.myrial.exceptions import *
import raco.algebra
import raco.expression
import raco.catalog
import raco.scheme
from raco.backends.myria import (MyriaLeftDeepTreeAlgebra,
                                 MyriaHyperCubeAlgebra,
                                 compile_to_json)
from raco.compile import optimize
from raco import relation_key
from raco.algebra import Shuffle

import collections
import copy
from functools import reduce


class DuplicateAliasException(Exception):

    """Bag comprehension arguments must have different alias names."""
    pass


class InvalidStatementException(Exception):
    pass


def get_unnamed_ref(column_ref, scheme, offset=0):
    """Convert a string or int into an attribute ref on the new table"""  # noqa
    if isinstance(column_ref, int):
        index = column_ref
    else:
        index = scheme.getPosition(column_ref)
    return raco.expression.UnnamedAttributeRef(index + offset)


def check_binop_compatability(op_name, left, right):
    """Check whether the arguments to an operation are compatible."""
    # Todo: check for type compatibility here?
    # https://github.com/uwescience/raco/issues/213
    if len(left.scheme()) != len(right.scheme()):
        raise SchemaMismatchException(op_name)


def check_assignment_compatability(before, after):
    """Check whether multiple assignments are compatible."""
    # TODO: check for exact schema match -- this is blocked by a general
    # cleanup of raco types.
    check_binop_compatability("assignment", before, after)


class ExpressionProcessor(object):

    """Convert syntactic expressions into relational algebra operations."""

    def __init__(self, symbols, catalog, use_dummy_schema=False):
        self.symbols = symbols
        self.catalog = catalog
        self.use_dummy_schema = use_dummy_schema

        # Variables accessed by the current operation
        self.uses_set = set()

    def get_and_clear_uses_set(self):
        """Retrieve the uses set and then clear its value."""
        try:
            return self.uses_set
        finally:
            self.uses_set = set()

    def evaluate(self, expr):
        method = getattr(self, expr[0].lower())
        return method(*expr[1:])

    def __lookup_symbol(self, _id):
        if _id not in self.symbols:
            raise NoSuchRelationException(_id)

        self.uses_set.add(_id)
        return copy.deepcopy(self.symbols[_id])

    def alias(self, _id):
        return self.__lookup_symbol(_id)

    def _get_scan_scheme(self, rel_key):
        try:
            return self.catalog.get_scheme(rel_key)
        except KeyError:
            if not self.use_dummy_schema:
                raise NoSuchRelationException(rel_key)
            # Create a dummy schema suitable for emitting plans
            return raco.scheme.DummyScheme()

    def scan(self, rel_key):
        """Scan a database table."""
        assert isinstance(rel_key, relation_key.RelationKey)
        scheme = self._get_scan_scheme(rel_key)
        return raco.algebra.Scan(rel_key, scheme,
                                 self.catalog.num_tuples(rel_key),
                                 self.catalog.partitioning(rel_key))

    def samplescan(self, rel_key, samp_size, is_pct, samp_type):
        """Sample a base relation."""
        assert isinstance(rel_key, relation_key.RelationKey)
        if samp_type not in ('WR', 'WoR'):
            raise MyrialCompileException(
                "Invalid Sampling Type: %s" % samp_type)
        scheme = self._get_scan_scheme(rel_key)
        return raco.algebra.SampleScan(rel_key, scheme, samp_size, is_pct,
                                       samp_type)

    def load(self, path, format, scheme, options):
        return raco.algebra.FileScan(path, format, scheme, options)

    def table(self, emit_clause):
        """Emit a single-row table literal."""
        emit_args = []
        for clause in emit_clause:
            emit_args.extend(clause.expand({}))

        from_args = collections.OrderedDict()
        from_args['$$SINGLETON$$'] = raco.algebra.SingletonRelation()

        # Add unbox relations to the from_args dictionary
        for name, sexpr in emit_args:
            self.extract_unbox_args(from_args, sexpr)

        op, info = multiway.merge(from_args)

        # rewrite clauses in terms of the new schema
        emit_args = [(name, multiway.rewrite_refs(sexpr, from_args, info))
                     for (name, sexpr) in emit_args]

        return raco.algebra.Apply(emitters=emit_args, input=op)

    @staticmethod
    def empty(_scheme):
        if not _scheme:
            _scheme = raco.scheme.Scheme()
        return raco.algebra.EmptyRelation(_scheme)

    def select(self, args):
        """Evaluate a select-from-where expression."""
        op = self.bagcomp(args.from_, args.where, args.select)
        if args.distinct:
            op = raco.algebra.Distinct(input=op)
        if args.limit is not None:
            op = raco.algebra.Limit(input=op, count=args.limit)

        return op

    def extract_unbox_args(self, from_args, sexpr):
        """Extract unbox arguments from a scalar expression.

        :param from_args: An ordered dictionary that maps from a
        relation alias (string) to an instance of raco.algebra.Operator.
        :param sexpr: A scalar expression (raco.expression.Expresssion)
        instance.
        """
        for sub_expr in sexpr.walk():
            if isinstance(sub_expr, raco.expression.Unbox):
                name = sub_expr.table_name
                assert isinstance(name, basestring)
                if name not in from_args:
                    from_args[name] = self.__lookup_symbol(name)

    def bagcomp(self, from_clause, where_clause, emit_clause):
        """Evaluate a bag comprehension.

        from_clause: A list of tuples of the form (id, expr).  expr can
        be None, which means "read the value from the symbol table".

        where_clause: An optional scalar expression (raco.expression).

        emit_clause: A list of EmitArg instances, each defining one or more
        output columns.
        """

        # Make sure no aliases were reused: [FROM X, X EMIT *] is illegal
        from_aliases = set([x[0] for x in from_clause])
        if len(from_aliases) != len(from_clause):
            raise DuplicateAliasException()

        # For each FROM argument, create a mapping from ID to operator
        # (id, raco.algebra.Operator)
        from_args = collections.OrderedDict()

        for _id, expr in from_clause:
            assert isinstance(_id, basestring)
            if expr:
                from_args[_id] = self.evaluate(expr)
            else:
                from_args[_id] = self.__lookup_symbol(_id)

        # Expand wildcards into a list of output columns
        assert emit_clause  # There should always be something to emit
        emit_args = []
        statemods = []
        for clause in emit_clause:
            emit_args.extend(clause.expand(from_args))
            statemods.extend(clause.get_statemods())

        orig_op, _info = multiway.merge(from_args)
        orig_schema_length = len(orig_op.scheme())

        # Add unbox relations to the from_args dictionary
        for name, sexpr in emit_args:
            self.extract_unbox_args(from_args, sexpr)
        if where_clause:
            self.extract_unbox_args(from_args, where_clause)

        # Create a single RA operation that is the cross of all targets
        op, info = multiway.merge(from_args)

        # HACK: calculate unboxed columns as implicit grouping columns,
        # so they can be used in grouping terms.
        new_schema_length = len(op.scheme())
        implicit_group_by_cols = range(orig_schema_length, new_schema_length)

        ################################################
        # Compile away unbox expressions in where, emit clauses
        ################################################

        if where_clause:
            where_clause = multiway.rewrite_refs(where_clause, from_args, info)
            # Extract the type of there where clause to force type safety
            # to be checked
            where_clause.typeof(op.scheme(), None)
            op = raco.algebra.Select(condition=where_clause, input=op)

        emit_args = [(name, multiway.rewrite_refs(sexpr, from_args, info))
                     for (name, sexpr) in emit_args]

        statemods = multiway.rewrite_statemods(statemods, from_args, info)

        if any(raco.expression.expression_contains_aggregate(ex)
               for name, ex in emit_args):
            return groupby.groupby(op, emit_args, implicit_group_by_cols,
                                   statemods)
        else:
            if statemods:
                return raco.algebra.StatefulApply(emit_args, statemods, op)
            if (len(from_args) == 1 and len(emit_clause) == 1 and
                isinstance(emit_clause[0],
                           (TableWildcardEmitArg, FullWildcardEmitArg))):
                return op
            return raco.algebra.Apply(emit_args, op)

    def distinct(self, expr):
        op = self.evaluate(expr)
        return raco.algebra.Distinct(input=op)

    def union(self, e1, e2):
        left = self.evaluate(e1)
        right = self.evaluate(e2)
        check_binop_compatability("union", left, right)
        return raco.algebra.Union(left, right)

    def unionall(self, e1):
        return raco.algebra.UnionAll([self.evaluate(e) for e in e1])

    def countall(self, expr):
        op = self.evaluate(expr)
        grouping_list = []
        agg_list = [raco.expression.COUNTALL()]
        return raco.algebra.GroupBy(grouping_list, agg_list, op)

    def intersect(self, e1, e2):
        left = self.evaluate(e1)
        right = self.evaluate(e2)
        check_binop_compatability("intersect", left, right)
        return raco.algebra.Intersection(left, right)

    def diff(self, e1, e2):
        left = self.evaluate(e1)
        right = self.evaluate(e2)
        check_binop_compatability("diff", left, right)
        return raco.algebra.Difference(left, right)

    def limit(self, expr, count):
        op = self.evaluate(expr)
        return raco.algebra.Limit(input=op, count=count)

    def cross(self, left_target, right_target):
        left = self.evaluate(left_target)
        right = self.evaluate(right_target)

        return raco.algebra.CrossProduct(left, right)

    def join(self, left_target, right_target):
        """Convert parser.JoinTarget arguments into a Join operation"""

        left = self.evaluate(left_target.expr)
        right = self.evaluate(right_target.expr)

        assert len(left_target.columns) == len(right_target.columns)

        left_scheme = left.scheme()
        left_refs = [get_unnamed_ref(c, left_scheme, 0)
                     for c in left_target.columns]

        right_scheme = right.scheme()
        right_refs = [get_unnamed_ref(c, right_scheme, len(left_scheme))
                      for c in right_target.columns]

        join_conditions = [raco.expression.EQ(x, y) for x, y in
                           zip(left_refs, right_refs)]

        # Merge the join conditions into a big AND expression

        def andify(x, y):
            """Merge two scalar expressions with an AND"""
            return raco.expression.AND(x, y)

        condition = reduce(andify, join_conditions)
        return raco.algebra.Join(condition, left, right)


class StatementProcessor(object):

    """Evaluate a list of statements"""

    def __init__(self, catalog, use_dummy_schema=False):
        # Map from identifiers (aliases) to raco.algebra.Operation instances
        self.symbols = {}

        assert isinstance(catalog, raco.catalog.Catalog)
        self.catalog = catalog

        self.ep = ExpressionProcessor(self.symbols, catalog, use_dummy_schema)

        self.cfg = ControlFlowGraph()

    def evaluate(self, statements):
        """Evaluate a list of statements"""
        for statement in statements:
            # Switch on the first tuple entry
            method = getattr(self, statement[0].lower())
            method(*statement[1:])

    def __evaluate_expr(self, expr, _def):
        """Evaluate an expression; add a node to the control flow graph.

        :param expr: An expression to evaluate
        :type expr: Myrial AST tuple
        :param _def: The variable defined by the expression, or None for
                     non-statements
        :type _def: string
        """

        op = self.ep.evaluate(expr)
        uses_set = self.ep.get_and_clear_uses_set()
        self.cfg.add_op(op, _def, uses_set)
        return op

    def __do_assignment(self, _id, expr):
        """Process an assignment statement; add a node to the control flow
        graph.
        :param _id: The target variable name.
        :type _id: string
        :param expr: The relational expression to evaluate
        :type expr: A Myrial expression AST node tuple
        """

        child_op = self.ep.evaluate(expr)
        if _id in self.symbols:
            check_assignment_compatability(child_op, self.symbols[_id])

        op = raco.algebra.StoreTemp(_id, child_op)
        uses_set = self.ep.get_and_clear_uses_set()
        self.cfg.add_op(op, _id, uses_set)

        # Point future references of this symbol to a scan of the materialized
        # table. Note that this assumes there is no scoping in Myrial.
        self.symbols[_id] = raco.algebra.ScanTemp(_id, child_op.scheme())

    def assign(self, _id, expr):
        """Map a variable to the value of an expression."""
        self.__do_assignment(_id, expr)

    def idbassign(self, _id, agg, expr):
        """Map an IDB to the value of an expression."""
        self.__do_assignment(_id, expr)

    def store(self, _id, rel_key, how_distributed):
        assert isinstance(rel_key, relation_key.RelationKey)

        alias_expr = ("ALIAS", _id)
        child_op = self.ep.evaluate(alias_expr)

        if how_distributed == "BROADCAST":
            child_op = raco.algebra.Broadcast(child_op)
        elif how_distributed == "ROUND_ROBIN":
            child_op = raco.algebra.Shuffle(
                child_op, None, shuffle_type=Shuffle.ShuffleType.RoundRobin)
        # hash-partitioned
        elif how_distributed:
            scheme = child_op.scheme()
            col_list = [get_unnamed_ref(a, scheme) for a in how_distributed]
            child_op = raco.algebra.Shuffle(
                child_op, col_list, shuffle_type=Shuffle.ShuffleType.Hash)
        op = raco.algebra.Store(rel_key, child_op)

        uses_set = self.ep.get_and_clear_uses_set()
        self.cfg.add_op(op, None, uses_set)

    def sink(self, _id):
        alias_expr = ("ALIAS", _id)
        child_op = self.ep.evaluate(alias_expr)
        op = raco.algebra.Sink(child_op)

        uses_set = self.ep.get_and_clear_uses_set()
        self.cfg.add_op(op, None, uses_set)

    def dump(self, _id):
        alias_expr = ("ALIAS", _id)
        child_op = self.ep.evaluate(alias_expr)
        op = raco.algebra.Dump(child_op)
        uses_set = self.ep.get_and_clear_uses_set()
        self.cfg.add_op(op, None, uses_set)

    def dowhile(self, statement_list, termination_ex):
        first_op_id = self.cfg.next_op_id  # op ID of the top of the loop

        for _type, _id, expr in statement_list:
            if _type != 'ASSIGN':
                # TODO: Better error message
                raise InvalidStatementException('%s not allowed in do/while' %
                                                _type.lower())
            self.__do_assignment(_id, expr)

        last_op_id = self.cfg.next_op_id

        self.__evaluate_expr(termination_ex, None)

        # Add a control flow edge from the loop condition to the top of the
        # loop
        self.cfg.add_edge(last_op_id, first_op_id)

    def check_schema(self, expr):
        return True

    def get_idb_leaves(self, expr, idbs):
        ret = []
        op = expr[0].lower()
        args = expr[1:]
        if op in ["bagcomp"]:
            for _id, arg in args[0]:
                if arg:
                    ret += self.get_idb_leaves(arg, idbs)
                elif _id in idbs:
                    ret += [_id]
        elif op in ["select"]:
            for _id, arg in args[0].from_:
                if arg:
                    ret += self.get_idb_leaves(arg, idbs)
                elif _id in idbs:
                    ret += [_id]
        elif op in ["join", "union", "cross", "diff", "intersect"]:
            ret += self.get_idb_leaves(args[0], idbs)
            ret += self.get_idb_leaves(args[1], idbs)
        elif op in ["unionall"]:
            for child in args[0]:
                ret += self.get_idb_leaves(child, idbs)
        elif op in ["limit", "countall", "distinct"]:
            ret += self.get_idb_leaves(args[0], idbs)
        elif op in ["alias"]:
            if args[0] in idbs:
                ret += [args[0]]
        else:
            raise InvalidStatementException('%s not recognized' % op)
        return ret

    def separate_inputs(self, expr, idbs, is_init):
        op = expr[0].lower()
        if op in ["unionall"]:
            inputs = []
            for input in expr[1]:
                inputs += self.separate_inputs(input, idbs, is_init)
            return inputs
        else:
            edb_only = len(self.get_idb_leaves(expr, idbs)) == 0
            if (is_init and edb_only) or (not is_init and not edb_only):
                return [expr]
        return []

    def untilconvergence(self, statement_list, recursion_mode,
                         pull_order_policy):
        idbs = {}
        idx = 0
        for _type, _id, emits, expr in statement_list:
            if _id in self.symbols:
                raise InvalidStatementException('IDB %s is already used' % _id)
            idbcontroller = raco.algebra.IDBController(
                _id, idx,
                [None, None, raco.algebra.EmptyRelation(raco.scheme.Scheme())],
                emits, None, recursion_mode)
            idbs[_id] = idbcontroller
            self.symbols[_id] = raco.algebra.ScanIDB(_id, None, idbcontroller)
            idx = idx + 1

        for _type, _id, emits, expr in statement_list:
            initial_inputs = self.separate_inputs(expr, idbs, True)
            if len(initial_inputs) == 0:
                idbs[_id].children()[0] =\
                    raco.algebra.EmptyRelation(raco.scheme.Scheme())
            elif len(initial_inputs) == 1:
                idbs[_id].children()[0] = self.ep.evaluate(initial_inputs[0])
            else:
                idbs[_id].children()[0] =\
                    raco.algebra.UnionAll([self.ep.evaluate(expr)
                                           for expr in initial_inputs])

        done = False
        while (not done):
            done = True
            for _type, _id, emits, expr in statement_list:
                if idbs[_id].children()[1] is not None:
                    continue
                leaves = self.get_idb_leaves(expr, idbs)
                if any(idbs[leaf].scheme() is None for leaf in leaves):
                    done = False
                else:
                    iterative_inputs = self.separate_inputs(expr, idbs, False)
                    if len(iterative_inputs) == 0:
                        idbs[_id].children()[1] = raco.algebra.EmptyRelation(
                            raco.scheme.Scheme())
                    elif len(iterative_inputs) == 1:
                        idbs[_id].children()[1] = self.ep.evaluate(
                            iterative_inputs[0])
                    else:
                        idbs[_id].children()[1] = raco.algebra.UnionAll(
                            [self.ep.evaluate(expr)
                             for expr in iterative_inputs])

        op = raco.algebra.UntilConvergence(idbs.values(), pull_order_policy)
        uses_set = self.ep.get_and_clear_uses_set()
        self.cfg.add_op(op, None, uses_set)

    def get_logical_plan(self, **kwargs):
        """Return an operator representing the logical query plan."""
        return self.cfg.get_logical_plan(
            dead_code_elimination=kwargs.get('dead_code_elimination', True),
            apply_chaining=kwargs.get('apply_chaining', True))

    def __get_physical_plan_for__(self, target_phys_algebra, **kwargs):
        logical_plan = self.get_logical_plan(**kwargs)

        kwargs['target'] = target_phys_algebra
        return optimize(logical_plan, **kwargs)

    def get_physical_plan(self, **kwargs):
        """Return an operator representing the physical query plan."""
        target_phys_algebra = kwargs.get('target_alg')
        if target_phys_algebra is None:
            if kwargs.get('multiway_join', False):
                target_phys_algebra = MyriaHyperCubeAlgebra(self.catalog)
            else:
                target_phys_algebra = MyriaLeftDeepTreeAlgebra()

        return self.__get_physical_plan_for__(target_phys_algebra, **kwargs)

    def get_json(self, **kwargs):
        lp = self.get_logical_plan()
        pps = self.get_physical_plan(**kwargs)

        # TODO This is not correct. The first argument is the raw query string,
        # not the string representation of the logical plan
        return compile_to_json(str(lp), pps, pps, "myrial")

    @classmethod
    def get_json_from_physical_plan(cls, pp):
        pps = pp

        # TODO This is not correct. The first argument is the raw query string,
        # not the string representation of the logical plan
        return compile_to_json(
            "NOT_SOURCED_FROM_LOGICAL_RA", pps, pps, "myrial")
