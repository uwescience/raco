#!/usr/bin/python

import raco.myrial.parser as parser
import raco.myrial.unbox as unbox
import raco.myrial.groupby as groupby
import raco.myrial.unpack_from as unpack_from
import raco.algebra
import raco.expression as sexpr
import raco.catalog
import raco.scheme

import collections
import random
import sys
import types

class DuplicateAliasException(Exception):
    """Bag comprehension arguments must have different alias names."""
    pass

class InvalidStatementException(Exception):
    pass

class ExpressionProcessor:
    '''Convert syntactic expressions into a relational algebra operation'''
    def __init__(self, symbols, db):
        self.symbols = symbols
        self.db = db

    def evaluate(self, expr):
        method = getattr(self, expr[0].lower())
        return method(*expr[1:])

    def alias(self, _id):
        return self.symbols[_id]

    def scan(self, relation_key, scheme):
        if not scheme:
            scheme = self.db.get_scheme(relation_key)

        rel = raco.catalog.Relation(relation_key, scheme)
        return raco.algebra.Scan(rel)

    def load(self, path, schema):
        raise NotImplementedError()

    def table(self, mappings):
        op = raco.algebra.SingletonRelation()
        return raco.algebra.Apply(mappings=mappings, input=op)

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
        # targets; re-write where and emit clauses to refer to its schema
        op, where_clause, emit_clause = unpack_from.unpack(
            from_args, where_clause, emit_clause)

        orig_scheme = op.scheme()
        op, where_clause, emit_clause = unbox.unbox(op, where_clause,
                                                    emit_clause, self.symbols)

        if where_clause:
            op = raco.algebra.Select(condition=where_clause, input=op)

        op, emit_clause = groupby.groupby(op, emit_clause)

        if emit_clause:
            op = raco.algebra.Apply(mappings=emit_clause, input=op)
        else:
            # Strip off any cross-product columns that we artificially added
            # during unboxing.
            mappings = [(orig_scheme.getName(i),
                         raco.expression.UnnamedAttributeRef(i))
                        for i in range(len(orig_scheme))]
            op = raco.algebra.Apply(mappings=mappings, input=op)

        return op

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
        # COUNT must take a column ref. Use the first column
        agg_list = [sexpr.COUNT(raco.expression.UnnamedAttributeRef(0))]
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

        condition = reduce(andify, join_conditions[1:], join_conditions[0])
        return raco.algebra.Join(condition, left, right)

class StatementProcessor:
    '''Evaluate a list of statements'''

    def __init__(self, db=None):
        # Map from identifiers (aliases) to raco.algebra.Operation instances
        self.symbols = {}

        # Identifiers that the user has asked us to materialize
        # (via store, dump, etc.).  Contains tuples of the form:
        # (id, raco.algebra.Operation)
        self.output_symbols = []

        self.db = db
        self.ep = ExpressionProcessor(self.symbols, db)

        # Create a unique prefix name for storing transient tables
        rnd  = str(random.randint(0,0x1000000000))
        self.transient_prefix = '__transient-' + rnd + "-"

    def evaluate(self, statements):
        '''Evaluate a list of statements'''
        for statement in statements:
            # Switch on the first tuple entry
            method = getattr(self, statement[0].lower())
            method(*statement[1:])

    def assign(self, _id, expr):
        '''Assign to a variable by modifying the symbol table'''
        # Evaluate the expression; store its result in a temporary table
        # TODO: Apply chaining when it is safe to do so
        # TODO: implement a leaf optimization to avoid duplicate
        # scan/insertions

        child_op = self.ep.evaluate(expr)
        key = self.transient_prefix + _id
        store_op = raco.algebra.Store(key, child_op)
        self.output_symbols.append((_id, store_op))

        # Point future references of this symbol to a scan of the
        # materialized table.
        # TODO: Make scan operate on the same relation key as store!
        relkey = raco.catalog.Relation(key, child_op.scheme())
        scan_op = raco.algebra.Scan(relkey)
        self.symbols[_id] = scan_op

    def store(self, _id, relation_key):
        child_op = self.symbols[_id]
        op = raco.algebra.Store(relation_key, child_op)
        self.output_symbols.append((_id, op))

    def dump(self, _id):
        child_op = self.symbols[_id]
        self.output_symbols.append((_id, child_op))

    @property
    def output_symbols(self):
        return self.output_symbols

    def explain(self, _id):
        raise NotImplementedError()

    def describe(self, _id):
        raise NotImplementedError()

    def dowhile(self, statement_list, termination_ex):
        body_ops = []
        for statement in statement_list:
            if statement[0] != 'ASSIGN':
                # TODO: Better error message
                raise InvalidStatementException('%s not allowed in do/while' %
                                                statement[0].lower())
            _id = statement[1]
            expr = statement[2]

            child_op = self.ep.evaluate(expr)
            key = self.transient_prefix + _id
            store_op = raco.algebra.Store(key, child_op)
            body_ops.append(store_op)

            relkey = raco.catalog.Relation(key, child_op.scheme())
            scan_op = raco.algebra.Scan(relkey)
            self.symbols[_id] = scan_op

        term_op = self.ep.evaluate(termination_ex)
        op = raco.algebra.DoWhile(body_ops, term_op)
        self.output_symbols.append(("do/while", op))
