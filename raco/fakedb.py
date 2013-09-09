
import collections
import itertools

import raco.algebra
import raco.scheme as scheme

class FakeDatabase:
    def __init__(self):
        # Map from relation names (strings) to tuples of (Bag, scheme.Scheme)
        self.tables = {}

    def evaluate(self, op):
        '''Evaluate a relational algebra operation.

        For "query-type" operators, return a tuple iterator.
        For store queries, the return value is None.
        '''
        method = getattr(self, op.opname().lower())
        return method(op)

    def evaluate_to_bag(self, op):
        '''Return a bag (collections.Counter instance) for the operation'''
        return collections.Counter(self.evaluate(op))

    def ingest(self, relation_key, contents, scheme):
        '''Directly load raw data into the database'''
        self.tables[relation_key] = (contents, scheme)

    def get_scheme(self, relation_key):
        bag, scheme = self.tables[relation_key]
        return scheme

    def scan(self, op):
        bag, scheme = self.tables[op.relation.name]
        return bag.elements()

    def select(self, op):
        child_it = self.evaluate(op.input)

        def filter_func(_tuple):
            # Note: this implicitly uses python truthiness rules for
            # interpreting non-boolean expressions.
            # TODO: Is this the the right semantics here?
            return op.condition.evaluate(_tuple, op.scheme())

        return itertools.ifilter(filter_func, child_it)
