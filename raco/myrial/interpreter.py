#!/usr/bin/python

import raco.myrial.parser as parser
import raco.algebra
import raco.expression as colexpr
import raco.catalog

import collections
import random
import sys
import types

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

    def table(self, tuple_list, schema):
        raise NotImplementedError()

    def bagcomp(self, from_expression, where_clause, emit_clause):
        # Evaluate the nested expression to get a RA operator
        op = self.evaluate(from_expression)

        if where_clause:
            op = raco.algebra.Select(condition=where_clause, input=op)

        if emit_clause:
            op = raco.algebra.Apply(mappings=emit_clause, input=op)

        return op

    def distinct(self, _id):
        # TODO: Use a first-class distinct operator here?
        op = self.symbols[_id]
        return raco.algebra.GroupBy(groupinglist=op.scheme().ascolumnlist(),
                                    input=op)

    def __process_bitop(self, _type, id1, id2):
        left = self.symbols[id1]
        right = self.symbols[id2]
        raise NotImplementedError()

    def unionall(self, id1, id2):
        left = self.symbols[id1]
        right = self.symbols[id2]
        # TODO: Figure out set/bag semantics here
        return raco.algebra.Union(left, right)

    def countall(self, _id):
        op = self.symbols[_id]
        return raco.algebra.GroupBy(groupinglist=[],
                                    aggregatelist=[colexpr.COUNT()],
                                    input=op)

    def intersect(self, id1, id2):
        raise NotImplementedError()

    def diff(self, id1, id2):
        raise NotImplementedError()

    def limit(self, _id, count):
        raise NotImplementedError()

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

        join_conditions = [colexpr.EQ(x,y) for x,y in
                           zip(left_refs, right_refs)]

        # Merge the join conditions into a big AND expression

        def andify(x,y):
            """Merge two column expressions with an AND"""
            return colexpr.AND(x,y)

        condition = reduce(andify, join_conditions[1:], join_conditions[0])
        return raco.algebra.Join(condition, left, right)

    def apply(self, _id, columns):
        op = self.symbols[_id]
        return raco.algebra.Apply(op, **columns)

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

    def evaluate(self, statements):
        '''Evaluate a list of statements'''
        for statement in statements:
            # Switch on the first tuple entry
            method = getattr(self, statement[0].lower())
            method(*statement[1:])

    def assign(self, _id, expr):
        '''Assign to a variable by modifying the symbol table'''
        op = self.ep.evaluate(expr)
        self.symbols[_id] = op

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
        pass

    def describe(self, _id):
        pass

    def dowhile(self, statement_list, termination_ex):
        pass
