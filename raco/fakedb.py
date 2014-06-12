
import collections
import itertools
import csv

from raco import relation_key
from raco import types

debug = False


class FakeDatabase(object):
    """An in-memory implementation of relational algebra operators"""

    def __init__(self):
        # Map from relation keys to tuples of (Bag, scheme.Scheme)
        self.tables = {}

        # Map from relation names to bags; schema is tracked by the runtime.
        self.temp_tables = {}

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

    def ingest(self, rel_key, contents, scheme):
        '''Directly load raw data into the database'''
        if isinstance(rel_key, str):
            rel_key = relation_key.RelationKey.from_string(rel_key)
        assert isinstance(rel_key, relation_key.RelationKey)
        self.tables[rel_key] = (contents, scheme)

    def get_scheme(self, rel_key):
        if isinstance(rel_key, str):
            rel_key = relation_key.RelationKey.from_string(rel_key)

        assert isinstance(rel_key, relation_key.RelationKey)

        (_, scheme) = self.tables[rel_key]
        return scheme

    def get_table(self, rel_key):
        """Retrieve the contents of table.

        :param rel_key: The key of the relation
        :type rel_key: relation_key.RelationKey
        :returns: A collections.Counter instance containing tuples.
        """
        if isinstance(rel_key, str):
            rel_key = relation_key.RelationKey.from_string(rel_key)
        assert isinstance(rel_key, relation_key.RelationKey)
        (contents, scheme) = self.tables[rel_key]
        return contents

    def get_temp_table(self, key):
        return self.temp_tables[key]

    def dump_all(self):
        for key, val in self.tables.iteritems():
            bag = val[0]
            print '%s: (%s)' % (key, bag)

        for key, bag in self.temp_tables.iteritems():
            print '__%s: (%s)' % (key, bag)

    def scan(self, op):
        assert isinstance(op.relation_key, relation_key.RelationKey)
        (bag, _) = self.tables[op.relation_key]
        return bag.elements()

    def filescan(self, op):
        type_list = op.scheme().get_types()

        with open(op.path, 'r') as fh:
            sample = fh.read(1024)
            dialect = csv.Sniffer().sniff(sample)
            fh.seek(0)
            reader = csv.reader(fh, dialect)
            for row in reader:
                pairs=  zip(row, type_list)
                cols = [types.parse_string(s, t) for s, t in pairs]
                yield tuple(cols)

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
                  for (_, colexpr) in op.emitters]
            return tuple(ls)
        return (make_tuple(t) for t in child_it)

    def statefulapply(self, op):
        child_it = self.evaluate(op.input)
        scheme = op.input.scheme()

        State = collections.namedtuple('State', ['scheme', 'values'])
        state = State(op.state_scheme, [expr.evaluate(None, scheme, None)
                      for (_, expr) in op.inits])

        def make_tuple(input_tuple, state):
            for i, new_state in enumerate(
                    [colexpr.evaluate(input_tuple, scheme, state)
                     for (_, colexpr) in op.updaters]):
                state.values[i] = new_state
            ls = [colexpr.evaluate(input_tuple, scheme, state)
                  for (_, colexpr) in op.emitters]
            return tuple(ls)
        return (make_tuple(t, state) for t in child_it)

    def join(self, op):
        # Compute the cross product of the children and flatten
        left_it = self.evaluate(op.left)
        right_it = self.evaluate(op.right)
        p1 = itertools.product(left_it, right_it)
        p2 = (x + y for (x, y) in p1)

        # Return tuples that match on the join conditions
        return (tpl for tpl in p2 if op.condition.evaluate(tpl, op.scheme()))

    def crossproduct(self, op):
        left_it = self.evaluate(op.left)
        right_it = self.evaluate(op.right)
        p1 = itertools.product(left_it, right_it)
        return (x + y for (x, y) in p1)

    def distinct(self, op):
        it = self.evaluate(op.input)
        s = set(it)
        return iter(s)

    def limit(self, op):
        it = self.evaluate(op.input)
        return itertools.islice(it, op.count)

    @staticmethod
    def singletonrelation(op):
        return iter([()])

    @staticmethod
    def emptyrelation(op):
        return iter([])

    def unionall(self, op):
        left_it = self.evaluate(op.left)
        right_it = self.evaluate(op.right)
        return itertools.chain(left_it, right_it)

    def difference(self, op):
        its = [self.evaluate(op.left), self.evaluate(op.right)]
        sets = [set(it) for it in its]
        return sets[0].difference(sets[1])

    def intersection(self, op):
        its = [self.evaluate(op.left), self.evaluate(op.right)]
        sets = [set(it) for it in its]
        return sets[0].intersection(sets[1])

    def groupby(self, op):
        child_it = self.evaluate(op.input)

        def process_grouping_columns(_tuple):
            ls = [sexpr.evaluate(_tuple, op.input.scheme()) for
                  sexpr in op.grouping_list]
            return tuple(ls)

        # Calculate groups of matching input tuples.
        # If there are no grouping terms, then all tuples are added
        # to a single bin.
        results = collections.defaultdict(list)

        if len(op.grouping_list) == 0:
            results[()] = list(child_it)
        else:
            for input_tuple in child_it:
                grouped_tuple = process_grouping_columns(input_tuple)
                results[grouped_tuple].append(input_tuple)

        # resolve aggregate functions
        for key, tuples in results.iteritems():
            agg_fields = [agg_expr.evaluate_aggregate(
                tuples, op.input.scheme()) for agg_expr in op.aggregate_list]
            yield(key + tuple(agg_fields))

    def sequence(self, op):
        for child_op in op.children():
            self.evaluate(child_op)
        return None

    def dowhile(self, op):
        i = 0

        children = op.children()
        body_ops = children[:-1]
        term_op = children[-1]

        if debug:
            print '---------- Values at top of do/while -----'
            self.dump_all()

        while True:
            for op in body_ops:
                self.evaluate(op)
            result_iterator = self.evaluate(term_op)

            if debug:
                i += 1
                print '-------- Iteration %d ------------' % i
                self.dump_all()

            try:
                tpl = result_iterator.next()

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
        assert isinstance(op.relation_key, relation_key.RelationKey)

        # Materialize the result
        bag = self.evaluate_to_bag(op.input)
        scheme = op.input.scheme()
        self.tables[op.relation_key] = (bag, scheme)
        return None

    def dump(self, op):
        for tpl in self.evaluate(op.input):
            print ','.join(tpl)
        return None

    def storetemp(self, op):
        bag = self.evaluate_to_bag(op.input)
        self.temp_tables[op.name] = bag

    def scantemp(self, op):
        bag = self.temp_tables[op.name]
        return bag.elements()

    def myriascan(self, op):
        return self.scan(op)

    def myriascantemp(self, op):
        return self.scantemp(op)

    def myriasymmetrichashjoin(self, op):
        it = self.join(op)

        # project-out columns
        def project(input_tuple):
            output = [input_tuple[x.position] for x in op.output_columns]
            return tuple(output)
        return (project(t) for t in it)

    def myriastore(self, op):
        return self.store(op)

    def myriastoretemp(self, op):
        return self.storetemp(op)

    def myriaapply(self, op):
        return self.apply(op)

    def myriastatefulapply(self, op):
        return self.statefulapply(op)

    def myriadupelim(self, op):
        return self.distinct(op)

    def myriaselect(self, op):
        return self.select(op)

    def myriacrossproduct(self, op):
        return self.crossproduct(op)

    def myriagroupby(self, op):
        return self.groupby(op)

    def myriashuffleconsumer(self, op):
        return self.evaluate(op.input)

    def myriashuffleproducer(self, op):
        return self.evaluate(op.input)

    def myriacollectconsumer(self, op):
        return self.evaluate(op.input)

    def myriacollectproducer(self, op):
        return self.evaluate(op.input)

    def myriabroadcastconsumer(self, op):
        return self.evaluate(op.input)

    def myriabroadcastproducer(self, op):
        return self.evaluate(op.input)

    def myriasingleton(self, op):
        return self.singletonrelation(op)

    def myriaemptyrelation(self, op):
        return self.emptyrelation(op)

    def myriaunionall(self, op):
        return self.unionall(op)

    def myriadifference(self, op):
        return self.difference(op)
