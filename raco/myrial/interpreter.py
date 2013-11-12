#!/usr/bin/python

import raco.myrial.parser as parser
import raco.myrial.groupby as groupby
import raco.myrial.unpack_from as unpack_from
import raco.algebra
import raco.expression as sexpr
import raco.catalog
import raco.scheme
from raco.language import MyriaAlgebra
from raco.algebra import LogicalAlgebra
from raco.myrialang import compile_to_json
from raco.compile import optimize

import collections
import random
import sys
import types

class DuplicateAliasException(Exception):
    """Bag comprehension arguments must have different alias names."""
    pass

class InvalidStatementException(Exception):
    pass

class NoSuchRelationException(Exception):
    pass

class UnboxState(object):
    """State maintained during the unbox phase of a bag comprehension."""
    def __init__(self, initial_pos):
        # This mapping keeps track of where unboxed expressions reside
        # in the output schema that is created from cross products of
        # all unboxed expressions.  The keys are relational expressions and
        # the values are tuples of the form: (operation, first_column_index)
        self.unbox_ops = collections.OrderedDict()

        # The next column index to be assigned
        self.pos = initial_pos

        # A set of column integers that are referenced by unbox operations
        self.column_refs = set()

class ExpressionProcessor:
    """Convert syntactic expressions into relational algebra operations."""
    def __init__(self, symbols, catalog, use_dummy_schema=False):
        self.symbols = symbols
        self.catalog = catalog
        self.use_dummy_schema = use_dummy_schema

    def evaluate(self, expr):
        method = getattr(self, expr[0].lower())
        return method(*expr[1:])

    def alias(self, _id):
        return self.symbols[_id]

    def scan(self, relation_key):
        """Scan a database table."""
        try:
            scheme = self.catalog.get_scheme(relation_key)
        except KeyError:
            if not self.use_dummy_schema:
                raise NoSuchRelationException(relation_key)

            # Create a dummy schema suitable for emitting plans
            scheme = raco.scheme.DummyScheme()

        return raco.algebra.Scan(relation_key, scheme)

    def load(self, path, schema):
        raise NotImplementedError()

    def __unbox_scalar_expression(self, sexpr, ub_state):
        def unbox_node(sexpr):
            if not isinstance(sexpr, raco.expression.Unbox):
                return sexpr
            else:
                rex = sexpr.relational_expression
                if not rex in ub_state.unbox_ops:
                    unbox_op = self.evaluate(rex)
                    scheme = unbox_op.scheme()
                    ub_state.unbox_ops[rex] = (unbox_op, ub_state.pos)
                    ub_state.pos += len(scheme)
                else:
                    unbox_op = ub_state.unbox_ops[rex][0]
                    scheme = unbox_op.scheme()

                offset = ub_state.unbox_ops[rex][1]
                if not sexpr.field:
                    # Default to column zero
                    pass
                elif type(sexpr.field) == types.IntType:
                    offset += sexpr.field
                else:
                    # resolve column name into a position
                    offset += scheme.getPosition(sexpr.field)

                ub_state.column_refs.add(offset)
                return raco.expression.UnnamedAttributeRef(offset)

        def recursive_eval(sexpr):
            """Apply unbox to a node and all its descendents."""
            newexpr = unbox_node(sexpr)
            newexpr.apply(recursive_eval)
            return newexpr

        return recursive_eval(sexpr)

    def __unbox(self, op, where_clause, emit_clause):
        """Apply unboxing to the clauses of a bag comprehension."""
        ub_state = UnboxState(len(op.scheme()))

        if where_clause:
            where_clause = self.__unbox_scalar_expression(where_clause,
                                                          ub_state)

        if emit_clause:
            emit_clause = [(name,
                            self.__unbox_scalar_expression(sexpr, ub_state))
                           for name, sexpr in emit_clause]

        def cross(x,y):
            return raco.algebra.CrossProduct(x,y)

        # Update the op to be the cross product of all unboxed relations
        cps = [v[0] for v in ub_state.unbox_ops.values()]
        op = reduce(cross, cps, op)
        return op, where_clause, emit_clause, ub_state.column_refs

    def __unbox_filter_group(self, op, where_clause, emit_clause):
        """Apply unboxing, filtering, and groupby."""

        # Record the original schema, so we can later strip off unboxed
        # columns.
        orig_scheme = op.scheme()
        op, where_clause, emit_clause, unbox_columns = self.__unbox(
            op, where_clause, emit_clause)

        if where_clause:
            op = raco.algebra.Select(condition=where_clause, input=op)

        if not emit_clause:
            # Strip off any columns that were added by unbox
            mappings = [(orig_scheme.getName(i),
                         raco.expression.UnnamedAttributeRef(i))
                        for i in range(len(orig_scheme))]
            return raco.algebra.Apply(mappings=mappings, input=op)
        else:
            # Apply any grouping operators
            return groupby.groupby(op, emit_clause, unbox_columns)

    def table(self, mappings):
        """Emit a single-row table literal."""
        op = raco.algebra.SingletonRelation()
        return self.__unbox_filter_group(op, None, mappings)

    def empty(self, _scheme):
        if not _scheme:
            _scheme = raco.scheme.Scheme()
        return raco.algebra.EmptyRelation(_scheme)

    def bagcomp(self, from_clause, where_clause, emit_clause):
        """Evaluate a bag comprehsion.

        from_clause: A list of tuples of the form (id, expr).  expr can
        be None, which means "read the value from the symbol table".

        where_clause: An optional scalar expression (raco.expression).

        emit_clause: An optional list of tuples of the form
        (column_name, scalar_expression).  The column name can be None, in
        which case the system concocts a column name.  If the emit_clause
        is None, all columns are emitted -- i.e., "EMIT *".
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
                from_args[_id] =  self.symbols[_id]

        # Create a single RA operation that is the rollup of all from
        # targets; re-write where and emit clauses to refer to its schema.
        op, where_clause, emit_clause = unpack_from.unpack(
            from_args, where_clause, emit_clause)

        return self.__unbox_filter_group(op, where_clause, emit_clause)

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
        agg_list = [sexpr.COUNTALL()]
        return raco.algebra.GroupBy(columnlist=grouping_list+agg_list,
                                    input=op)

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

        join_conditions = [sexpr.EQ(x,y) for x,y in
                           zip(left_refs, right_refs)]

        # Merge the join conditions into a big AND expression

        def andify(x,y):
            """Merge two scalar expressions with an AND"""
            return sexpr.AND(x,y)

        condition = reduce(andify, join_conditions)
        return raco.algebra.Join(condition, left, right)

class StatementProcessor:
    '''Evaluate a list of statements'''

    def __init__(self, catalog=None, use_dummy_scheme=False):
        # Map from identifiers (aliases) to raco.algebra.Operation instances
        self.symbols = {}

        # A sequence of plans to be executed by the database
        self.output_ops = []

        self.catalog = catalog
        self.ep = ExpressionProcessor(self.symbols, catalog, use_dummy_scheme)

        # Unique identifiers for temporary tables created by DUMP operations
        self.dump_output_id = 0

    def evaluate(self, statements):
        '''Evaluate a list of statements'''
        for statement in statements:
            # Switch on the first tuple entry
            method = getattr(self, statement[0].lower())
            method(*statement[1:])

    def __materialize_result(self, _id, expr, op_list):
        '''Materialize an expression as a temporary table.'''
        child_op = self.ep.evaluate(expr)
        store_op = raco.algebra.StoreTemp(_id, child_op)
        op_list.append(store_op)

        # Point future references of this symbol to a scan of the
        # materialized table.
        self.symbols[_id] = raco.algebra.ScanTemp(_id, child_op.scheme())

    def assign(self, _id, expr):
        '''Map a variable to the value of an expression.'''

        # TODO: Apply chaining when it is safe to do so
        # TODO: implement a leaf optimization to avoid duplicate
        # scan/insertions
        self.__materialize_result(_id, expr, self.output_ops)

    def store(self, _id, relation_key):
        child_op = self.symbols[_id]
        op = raco.algebra.Store(relation_key, child_op)
        self.output_ops.append(op)

    def dump(self, _id):
        child_op = self.symbols[_id]
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
        lp_str = str(lp)
        pps = optimize([('root', lp)], target=MyriaAlgebra,
                       source=LogicalAlgebra)
        return compile_to_json(lp, lp, pps)

    def explain(self, _id):
        raise NotImplementedError()

    def describe(self, _id):
        raise NotImplementedError()

    def dowhile(self, statement_list, termination_ex):
        body_ops = []
        for _type, _id, expr in statement_list:
            if _type != 'ASSIGN':
                # TODO: Better error message
                raise InvalidStatementException('%s not allowed in do/while' %
                                                _type.lower())
            self.__materialize_result(_id, expr, body_ops)

        term_op = self.ep.evaluate(termination_ex)
        op = raco.algebra.DoWhile(raco.algebra.Sequence(body_ops), term_op)
        self.output_ops.append(op)
