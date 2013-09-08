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
    def __init__(self, symbols):
        self.symbols = symbols

    def evaluate(self, expr):
        method = getattr(self, expr[0].lower())
        return method(*expr[1:])

    def alias(self, _id):
        return self.symbols[_id]

    def scan(self, relation_key, scheme):
        # TODO(AJW) resolve the schema if it's not provided
        assert scheme # REMOVE THIS!

        rel = raco.catalog.Relation(relation_key, scheme)
        return raco.algebra.Scan(rel)

    def load(self, path, schema):
        raise NotImplementedError()

    def table(self, tuple_list, schema):
        raise NotImplementedError()

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

    def join(self, arg1, arg2):
        # Note: arguments are of type parser.JoinTarget
        left = self.symbols[arg1.id]
        right = self.symbols[arg2.id]

        assert len(arg1.columns) == len(arg2.columns)

        condition = [colexpr.EQ(x,y) for x,y in zip(arg1.columns, arg2.columns)]
        return raco.algebra.Join(condition, left, right)

    def apply(self, _id, columns):
        op = self.symbols[_id]
        return raco.algebra.Apply(op, **columns)

class StatementProcessor:
    '''Evaluate a list of statements'''

    def __init__(self):
        # Map from identifiers (aliases) to raco.algebra.Operation instances
        self.symbols = {}

        # Identifiers that the user has asked us to materialize
        # (via store, dump, etc.).  Contains tuples of the form:
        # (id, raco.algebra.Operation)
        self.output_symbols = []

        self.ep = ExpressionProcessor(self.symbols)

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
        self.output_symbols.append(_id, op)

    @property
    def output_symbols(self):
        return self.output_symbols

    def explain(self, _id):
        pass

    def describe(self, _id):
        pass

    def dump(self, _id):
        pass

    def dowhile(self, statement_list, termination_ex):
        pass
