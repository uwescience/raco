
import collections
import itertools

import raco.algebra
import raco.scheme

debug = False

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

    def dump_all(self):
        for key, val in self.tables.iteritems():
            bag = val[0]
            print '%s: (%s)' % (key, bag)

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
        scheme = op.input.scheme()

        def make_tuple(input_tuple):
            ls = [colexpr.evaluate(input_tuple, scheme)
                  for var, colexpr in op.mappings]
            return tuple(ls)
        return (make_tuple(t) for t in child_it)

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

    def singletonrelation(self, op):
        return iter([()])

    def unionall(self, op):
        left_it = self.evaluate(op.left)
        right_it = self.evaluate(op.right)
        return itertools.chain(left_it, right_it)

    def difference(self, op):
        its = [self.evaluate(op.left), self.evaluate(op.right)]
        bags = [collections.Counter(it) for it in its]
        return (bags[0] - bags[1]).elements()

    def intersection(self, op):
        its = [self.evaluate(op.left), self.evaluate(op.right)]
        bags = [collections.Counter(it) for it in its]
        return (bags[0] & bags[1]).elements()

    def groupby(self, op):
        child_it = self.evaluate(op.input)

        def process_grouping_columns(_tuple):
            ls = [sexpr.evaluate(_tuple, op.input.scheme()) for
                  sexpr in op.groupinglist]
            return tuple(ls)

        # Calculate groups of matching input tuples.
        # If there are no grouping terms, then all tuples are added
        # to a single bin.
        results = collections.defaultdict(list)

        if len(op.groupinglist) == 0:
            results[()] = list(child_it)
        else:
            for input_tuple in child_it:
                grouped_tuple = process_grouping_columns(input_tuple)
                results[grouped_tuple].append(input_tuple)

        # resolve aggregate functions
        for key, tuples in results.iteritems():
            agg_fields = [agg_expr.evaluate_aggregate(
                tuples, op.input.scheme()) for agg_expr in op.aggregatelist]
            yield(key + tuple(agg_fields))


    def dowhile(self, op):
        i = 0
        while True:
            for body_op in op.body_ops:
                if debug:
                    print 'Evaluating: %s' % str(body_op)
                self.evaluate(body_op)

            result = self.evaluate(op.term_op)

            if debug:
                i += 1
                print '-------- Iteration %d ------------' % i
                self.dump_all()

            try:
                tpl = result.next()

                if debug:
                    print 'Term: %s' % str(tpl)

                # XXX should we use python truthiness here?
                if not tpl[0]:
                    break
            except StopIteration:
                break
            except IndexError:
                break

    def store(self, op):
        # Materialize the result
        bag = self.evaluate_to_bag(op.input)
        scheme = op.input.scheme()
        self.tables[op.name] = (bag, scheme)
        return None
