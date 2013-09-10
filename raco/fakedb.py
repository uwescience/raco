
import collections
import itertools

import raco.algebra
import raco.scheme as scheme

class FakeDatabase:
    """An in-memory implementation of relational algebra operators"""

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

    def apply(self, op):
        child_it = self.evaluate(op.input)
        colexprs = [c for _, c in op.mappings]

        def make_tuple(input_tuple):
            """Convert an input tuple into an output tuple.

            Note: aggregate functions are not yet evaluated
            """
            ls = [colexpr.evaluate(input_tuple, op.input.scheme()) \
                  for colexpr in colexprs]
            return tuple(ls)

        if op.set_semantics:
            # Calculate groups of matching input tuples
            results = collections.defaultdict(list)
            for input_tuple in child_it:
                output_tuple = make_tuple(input_tuple)
                results[output_tuple].append(input_tuple)

            # Resolve aggregate functions
            for key, tuples in results.iteritems():
                output_fields = list(key)
                for i in range(len(colexprs)):
                    if isinstance(colexprs[i],
                                  raco.expression.AggregateExpression):
                        output_fields[i] = colexprs[i].evaluate_aggregate(
                            tuples, op.input.scheme())
                    else:
                        output_fields[i] = key[i]
                yield tuple(output_fields)
        else:
            for t in child_it:
                yield make_tuple(t)

    def join(self, op):
        # Compute the cross product of the children and flatten
        left_it = self.evaluate(op.left)
        right_it = self.evaluate(op.right)
        p1 = itertools.product(left_it, right_it)
        p2 = (x + y for (x,y) in p1)

        # Return tuples that match on the join conditions
        return (tpl for tpl in p2 if op.condition.evaluate(tpl, op.scheme()))

    def crossproduct(self, op):
        left_it = self.evaluate(op.left)
        right_it = self.evaluate(op.right)
        p1 = itertools.product(left_it, right_it)
        return (x + y for (x,y) in p1)

    def distinct(self, op):
        it = self.evaluate(op.input)
        s = set(it)
        return iter(s)

    def limit(self, op):
        it = self.evaluate(op.input)
        return itertools.islice(it, op.count)
