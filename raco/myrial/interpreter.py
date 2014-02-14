import raco.myrial.groupby as groupby
import raco.myrial.multiway as multiway
import raco.algebra
import raco.expression
import raco.catalog
import raco.scheme
from raco.language import MyriaAlgebra
from raco.algebra import LogicalAlgebra
from raco.myrialang import compile_to_json
from raco.compile import optimize
from raco import relation_key

import collections
import types
import copy
import networkx as nx

class DuplicateAliasException(Exception):
    """Bag comprehension arguments must have different alias names."""
    pass

class InvalidStatementException(Exception):
    pass

class NoSuchRelationException(Exception):
    pass

def lookup_symbol(symbols, _id):
    return copy.copy(symbols[_id])

class ExpressionProcessor(object):
    """Convert syntactic expressions into relational algebra operations."""
    def __init__(self, symbols, catalog, use_dummy_schema=False):
        self.symbols = symbols
        self.catalog = catalog
        self.use_dummy_schema = use_dummy_schema

        # Variables accesed by the current operation
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

    def alias(self, _id):
        self.uses_set.add(_id)
        return lookup_symbol(self.symbols, _id)

    def scan(self, rel_key):
        """Scan a database table."""
        assert isinstance(rel_key, relation_key.RelationKey)
        try:
            scheme = self.catalog.get_scheme(rel_key)
        except KeyError:
            if not self.use_dummy_schema:
                raise NoSuchRelationException(rel_key)

            # Create a dummy schema suitable for emitting plans
            scheme = raco.scheme.DummyScheme()

        return raco.algebra.Scan(rel_key, scheme)

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
        for sub_expr in sexpr.walk():
            if isinstance(sub_expr, raco.expression.Unbox):
                rex = sub_expr.relational_expression
                if not rex in from_args:
                    unbox_op = self.evaluate(rex)
                    from_args[rex] = unbox_op

    def bagcomp(self, from_clause, where_clause, emit_clause):
        """Evaluate a bag comprehsion.

        from_clause: A list of tuples of the form (id, expr).  expr can
        be None, which means "read the value from the symbol table".

        where_clause: An optional scalar expression (raco.expression).

        emit_clause: A list of EmitArg instances, each defining one or more
        output columns.
        """

        # Make sure no aliases were reused: [FROM X, X EMIT *] is illegal
        from_aliases = set([x[0] for x in from_clause])
        if len(from_aliases) != len(from_clause):
            raise DuplicateAliasException();

        # For each FROM argument, create a mapping from ID to operator
        # (id, raco.algebra.Operator)
        from_args = collections.OrderedDict()

        for _id, expr in from_clause:
            if expr:
                from_args[_id] =  self.evaluate(expr)
            else:
                from_args[_id] = lookup_symbol(self.symbols, _id)

        # Expand wildcards into a list of output columns
        assert emit_clause # There should always be something to emit
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

        # rewrite clauses in terms of the new schema
        if where_clause:
            where_clause = multiway.rewrite_refs(where_clause, from_args, info)
            op = raco.algebra.Select(condition=where_clause, input=op)

        emit_args = [(name, multiway.rewrite_refs(sexpr, from_args, info))
                      for (name, sexpr) in emit_args]

        statemods = [(name, init, multiway.rewrite_refs(update, from_args, info))
                    for name, init, update in statemods]

        if any([raco.expression.isaggregate(ex) for name, ex in emit_args]):
            return groupby.groupby(op, emit_args, implicit_group_by_cols)
        else:
            return raco.algebra.StatefulApply(emit_args, statemods, op)

    def distinct(self, expr):
        op = self.evaluate(expr)
        return raco.algebra.Distinct(input=op)

    def unionall(self, e1, e2):
        left = self.evaluate(e1)
        right = self.evaluate(e2)
        return raco.algebra.UnionAll(left, right)

    def countall(self, expr):
        op = self.evaluate(expr)
        grouping_list = []
        agg_list = [raco.expression.COUNTALL()]
        return raco.algebra.GroupBy(grouping_list, agg_list, op)

    def intersect(self, e1, e2):
        left = self.evaluate(e1)
        right = self.evaluate(e2)
        return raco.algebra.Intersection(left, right)

    def diff(self, e1, e2):
        left = self.evaluate(e1)
        right = self.evaluate(e2)
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

        def get_attribute_ref(column_ref, scheme, offset):
            """Convert a string or int into an attribute ref on the new table"""
            if type(column_ref) == types.IntType:
                index = column_ref
            else:
                index = scheme.getPosition(column_ref)
            return raco.expression.UnnamedAttributeRef(index + offset)

        left_scheme = left.scheme()
        left_refs = [get_attribute_ref(c, left_scheme, 0)
                     for c in left_target.columns]

        right_scheme = right.scheme()
        right_refs = [get_attribute_ref(c, right_scheme, len(left_scheme))
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
    '''Evaluate a list of statements'''

    def __init__(self, catalog=None, use_dummy_schema=False):
        # Map from identifiers (aliases) to raco.algebra.Operation instances
        self.symbols = {}

        # A sequence of plans to be executed by the database
        self.output_ops = []

        self.catalog = catalog
        self.ep = ExpressionProcessor(self.symbols, catalog, use_dummy_schema)

        # Unique identifiers for temporary tables created by DUMP operations
        self.dump_output_id = 0

        # Control flow graph: nodes are operations, edges are control flow
        self.cfg = nx.DiGraph()

        # Unique identifiers for operation IDs
        self.next_op_id = 0

    def evaluate(self, statements):
        '''Evaluate a list of statements'''
        for statement in statements:
            # Switch on the first tuple entry
            method = getattr(self, statement[0].lower())
            method(*statement[1:])

    def __evaluate_expr(self, expr, def_set):
        """Evaluate an expression; add a node to the control flow graph."""

        op_id = self.next_op_id
        self.next_op_id += 1

        op_out = self.ep.evaluate(expr)
        uses_set = self.ep.get_and_clear_uses_set()
        self.cfg.add_node(op_id, defs=def_set, uses=uses_set)

        # Add a control flow edge from the prevoius statement; this assumes we
        # don't do jumps or any other non-linear control flow.
        if op_id > 0:
            self.cfg.add_edge(op_id - 1, op_id)
        return op_out

    def __do_assignment(self, _id, expr, op_list):
        """Process an assignment statement.

        :param _id: The target variable name.
        :type _id: string
        :param expr: The relational expression to evaluate
        :type expr: A Myrial expression AST node tuple
        :param op_list: A list of output operations to capture the Store operation
        """

        child_op = self.__evaluate_expr(expr, {_id})

        # Wrap the output of the operation in a store to a temporary variable so
        # we can later retrieve its value
        store_op = raco.algebra.StoreTemp(_id, child_op)
        op_list.append(store_op)

        # Point future references of this symbol to a scan of the
        # materialized table. Note that this assumes there is no scoping in Myrial.
        self.symbols[_id] = raco.algebra.ScanTemp(_id, child_op.scheme())

    def assign(self, _id, expr):
        '''Map a variable to the value of an expression.'''
        self.__do_assignment(_id, expr, self.output_ops)

    def store(self, _id, rel_key):
        assert isinstance(rel_key, relation_key.RelationKey)
        alias_expr = ("ALIAS", _id)
        child_op = self.__evaluate_expr(alias_expr, set())
        op = raco.algebra.Store(rel_key, child_op)
        self.output_ops.append(op)

    def dump(self, _id):
        alias_expr = ("ALIAS", _id)
        child_op = self.__evaluate_expr(alias_expr, set())
        op = raco.algebra.StoreTemp("__OUTPUT%d__" % self.dump_output_id,
                                    child_op)
        self.dump_output_id += 1
        self.output_ops.append(op)

    def get_logical_plan(self):
        """Return an operator representing the logical query plan."""
        return raco.algebra.Sequence(self.output_ops)

    def get_physical_plan(self):
        """Return an operator representing the physical query plan."""

        # TODO: Get rid of the dummy label argument here.
        # Return first (only) plan; strip off dummy label.
        logical_plan = self.get_logical_plan()
        physical_plans = optimize([('root', logical_plan)],
                                  target=MyriaAlgebra,
                                  source=LogicalAlgebra)
        return physical_plans[0][1]

    def get_json(self):
        lp = self.get_logical_plan()
        pps = optimize([('root', lp)], target=MyriaAlgebra,
                       source=LogicalAlgebra)
        # TODO This is not correct. The first argument is the raw query string,
        # not the string representation of the logical plan
        return compile_to_json(str(lp), lp, pps)

    def dowhile(self, statement_list, termination_ex):
        body_ops = []
        first_op_id = self.next_op_id # op ID of the top of the loop

        for _type, _id, expr in statement_list:
            if _type != 'ASSIGN':
                # TODO: Better error message
                raise InvalidStatementException('%s not allowed in do/while' %
                                                _type.lower())
            self.__do_assignment(_id, expr, body_ops)

        last_op_id = self.next_op_id

        term_op = self.__evaluate_expr(termination_ex, set())
        op = raco.algebra.DoWhile(raco.algebra.Sequence(body_ops), term_op)
        self.output_ops.append(op)

        # Add a control flow edge from the loop condition to the top of the loop
        self.cfg.add_edge(last_op_id, first_op_id)
