
import collections
import itertools
import csv
import random

from raco.dbconn import DBConnection
from raco import relation_key, types
from raco.algebra import StoreTemp, DEFAULT_CARDINALITY
from raco.catalog import Catalog
from raco.expression import AND, EQ, BuiltinAggregateExpression
from raco.representation import RepresentationProperties

debug = False


class State(object):
    def __init__(self, op_scheme, state_scheme, init_exprs):
        self.scheme = state_scheme
        self.op_scheme = op_scheme
        self.values = [x.evaluate(None, op_scheme, None)
                       for (_, x) in init_exprs]

    def update(self, tpl, update_exprs):
        new_vals = [expr.evaluate(tpl, self.op_scheme, self)
                    for (_, expr) in update_exprs]
        self.values = new_vals

    def __str__(self):
        return 'State(%s)' % self.values


class FakeDatabase(Catalog):
    """An in-memory implementation of relational algebra operators"""

    def __init__(self):
        # Persistent tables, identified by RelationKey
        self.tables = DBConnection()

        # Temporary tables, identified by string name
        self.temp_tables = DBConnection()

        # partitionings
        self.partitionings = {}

    def get_num_servers(self):
        return 1

    def num_tuples(self, rel_key):
        try:
            return self.tables.num_tuples(rel_key)
        except KeyError:
            return DEFAULT_CARDINALITY

    def partitioning(self, rel_key):
        """get fake metadata for relation.
        This has no effect on query evaluation
        in the FakeDatabase"""
        return self.partitionings[rel_key]

    def evaluate(self, op):
        """Evaluate a relational algebra operation.

        For "query-type" operators, return a tuple iterator.
        For store queries, the return value is None.
        """
        method = getattr(self, op.opname().lower())
        return method(op)

    def evaluate_to_bag(self, op):
        """Return a bag (collections.Counter instance) for the operation"""
        return collections.Counter(self.evaluate(op))

    def ingest(self, rel_key, contents, scheme,
               partitioning=RepresentationProperties()):
        """Directly load raw data into the database"""
        if isinstance(rel_key, basestring):
            rel_key = relation_key.RelationKey.from_string(rel_key)
        assert isinstance(rel_key, relation_key.RelationKey)
        self.tables.add_table(rel_key, scheme, contents.elements())
        self.partitionings[rel_key] = partitioning

    def add_function(self, tup):
        print ("added function")
        return self.tables.register_function(tup)

    def get_function(self, name):
        if name == "":
            raise ValueError("Invalid UDF name.")
        return self.tables.get_function(name)

    def get_scheme(self, rel_key):
        if isinstance(rel_key, basestring):
            rel_key = relation_key.RelationKey.from_string(rel_key)

        assert isinstance(rel_key, relation_key.RelationKey)

        return self.tables.get_scheme(rel_key)

    def get_table(self, rel_key):
        """Retrieve the contents of table.

        :param rel_key: The key of the relation
        :type rel_key: relation_key.RelationKey
        :returns: A collections.Counter instance containing tuples.
        """
        if isinstance(rel_key, basestring):
            rel_key = relation_key.RelationKey.from_string(rel_key)
        assert isinstance(rel_key, relation_key.RelationKey)
        return self.tables.get_table(rel_key)

    def get_temp_table(self, key):
        return self.temp_tables.get_table(key)

    def delete_temp_table(self, key):
        self.temp_tables.delete_table(key)

    def dump_all(self):
        for key, val in self.tables.iteritems():
            bag = val[0]
            print '%s: (%s)' % (key, bag)

        for key, bag in self.temp_tables.iteritems():
            print '__%s: (%s)' % (key, bag)

    def scan(self, op):
        assert isinstance(op.relation_key, relation_key.RelationKey)
        return self.tables.get_table(op.relation_key).elements()

    def calculatesamplingdistribution(self, op):
        if op.is_pct:
            tup_cnt = sum(t[1] for t in list(self.evaluate(op.input)))
            sample_size = int(round(tup_cnt * (op.sample_size / 100.0)))
        else:
            sample_size = op.sample_size
        return (t + (sample_size, op.sample_type) for t in
                self.evaluate(op.input))

    def sample(self, op):
        sample_info = list(self.evaluate(op.left))
        assert len(sample_info) == 1
        sample_type = sample_info[0][3]
        sample_size = sample_info[0][2]
        tuples = list(self.evaluate(op.right))
        if sample_type == 'WR':
            # Add unique index to make them appear like different tuples.
            sample = [(i,) + random.choice(tuples) for i in range(sample_size)]
        elif sample_type == 'WoR':
            sample = random.sample(tuples, sample_size)
        else:
            raise ValueError("Invalid sample type")
        return iter(sample)

    def filescan(self, op):
        type_list = op.scheme().get_types()

        with open(op.path, 'r') as fh:
            if not op.options:
                sample = fh.read(1024)
                dialect = csv.Sniffer().sniff(sample)
                fh.seek(0)
                reader = csv.reader(fh, dialect)
            else:
                options = {
                    'delimiter': ",",
                    'quote': '"',
                    'escape': None,
                    'skip': 0}
                options.update(op.options)
                reader = csv.reader(
                    fh,
                    delimiter=options['delimiter'],
                    quotechar=options['quote'],
                    escapechar=options['escape'])
                if options['skip']:
                    for _ in xrange(options['skip']):
                        next(fh)
            for row in reader:
                pairs = zip(row, type_list)
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

        state = State(scheme, op.state_scheme, op.inits)

        def make_tuple(input_tuple, state):
            # Update state variables
            state.update(input_tuple, op.updaters)

            # Extract a result for each emit expression
            return tuple([colexpr.evaluate(input_tuple, scheme, state)
                          for (_, colexpr) in op.emitters])

        return (make_tuple(t, state) for t in child_it)

    def join(self, op):
        # Compute the cross product of the children and flatten
        left_it = self.evaluate(op.left)
        right_it = self.evaluate(op.right)
        p1 = itertools.product(left_it, right_it)
        p2 = (x + y for (x, y) in p1)

        # Return tuples that match on the join conditions
        return (tpl for tpl in p2 if op.condition.evaluate(tpl, op.scheme()))

    def projectingjoin(self, op):
        # standard join, projecting the output columns
        return (tuple(t[x.position] for x in op.output_columns)
                for t in self.join(op))

    def naryjoin(self, op):
        def eval_conditions(conditions, tpl):
            """Turns the weird NaryJoin condition set into a proper
            expression, then evaluates it."""
            cond = reduce(lambda a, b: AND(a, b),
                          map(lambda (a, b): EQ(a, b), conditions))
            return cond.evaluate(tpl, op.scheme())

        # Elements of prod are tuples of tuples like ((1, 2), (3, 4))
        prod = itertools.product(*(self.evaluate(child)
                                   for child in op.children()))
        # Elements of tuples have been flattened like (1, 2, 3, 4)
        tuples = (sum(x, ()) for x in prod)
        return (tpl for tpl in tuples if eval_conditions(op.conditions, tpl))

    def crossproduct(self, op):
        left_it = self.evaluate(op.left)
        right_it = self.evaluate(op.right)
        p1 = itertools.product(left_it, right_it)
        return (x + y for (x, y) in p1)

    def distinct(self, op):
        it = self.evaluate(op.input)
        s = set(it)
        return iter(s)

    def project(self, op):
        if not op.columnlist:
            return self.distinct(op)

        return set(tuple(t[x.position] for x in op.columnlist)
                   for t in self.evaluate(op.input))

    def limit(self, op):
        it = self.evaluate(op.input)
        return itertools.islice(it, op.count)

    @staticmethod
    def singletonrelation(op):
        return iter([()])

    @staticmethod
    def emptyrelation(op):
        return iter([])

    def union(self, op):
        return set(self.evaluate(op.left)).union(set(self.evaluate(op.right)))

    def unionall(self, op):
        return itertools.chain.from_iterable(
            self.evaluate(arg) for arg in op.args)

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
        input_scheme = op.input.scheme()

        def process_grouping_columns(_tuple):
            ls = [sexpr.evaluate(_tuple, input_scheme) for
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
            state = State(input_scheme, op.state_scheme, op.inits)
            for tpl in tuples:
                state.update(tpl, op.updaters)

            # For now, built-in aggregates are handled differently than UDA
            # aggregates.  TODO: clean this up!

            agg_fields = []
            for expr in op.aggregate_list:
                if isinstance(expr, BuiltinAggregateExpression):
                    # Old-style aggregate: pass all tuples to the eval func
                    agg_fields.append(
                        expr.evaluate_aggregate(tuples, input_scheme))
                else:
                    # UDA-style aggregate: evaluate a normal expression that
                    # can reference only the state tuple
                    agg_fields.append(expr.evaluate(None, None, state))
            yield(key + tuple(agg_fields))

    def sequence(self, op):
        for child_op in op.children():
            self.evaluate(child_op)
        return None

    def parallel(self, op):
        for child_op in op.children():
            self.evaluate(child_op)
        return None

    def dowhile(self, op):
        i = 0

        children = op.children()
        body_ops = children[:-1]
        term_op = children[-1]
        if isinstance(term_op, StoreTemp):
            term_op = term_op.input

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

    def debroadcast(self, op):
        return self.evaluate(op.input)

    def store(self, op):
        assert isinstance(op.relation_key, relation_key.RelationKey)

        scheme = op.input.scheme()
        self.tables.add_table(op.relation_key, scheme, self.evaluate(op.input))
        return None

    def sink(self, op):
        scheme = op.input.scheme()
        self.tables.add_table(
            relation_key.RelationKey("OUTPUT"),
            scheme, self.evaluate(op.input))
        return None

    def dump(self, op):
        for tpl in self.evaluate(op.input):
            print ','.join(tpl)
        return None

    def storetemp(self, op):
        scheme = op.input.scheme()
        self.temp_tables.add_table(op.name, scheme, self.evaluate(op.input))

    def appendtemp(self, op):
        self.temp_tables.append_table(op.name, self.evaluate(op.input))

    def scantemp(self, op):
        return self.temp_tables.get_table(op.name).elements()

    def myriascan(self, op):
        return self.scan(op)

    def myriacalculatesamplingdistribution(self, op):
        return self.calculatesamplingdistribution(op)

    def myriasample(self, op):
        return self.sample(op)

    def myriafilescan(self, op):
        return self.filescan(op)

    def myriasink(self, op):
        return self.sink(op)

    def myriascantemp(self, op):
        return self.scantemp(op)

    def myrialimit(self, op):
        return self.limit(op)

    def myriasymmetrichashjoin(self, op):
        return self.projectingjoin(op)

    def myrialeapfrogjoin(self, op):
        # standard naryjoin, projecting the output columns
        return (tuple(t[x.position] for x in op.output_columns)
                for t in self.naryjoin(op))

    def myriainmemoryorderby(self, op):
        return self.evaluate(op.input)

    def myriahypercubeshuffleconsumer(self, op):
        return self.evaluate(op.input)

    def myriahypercubeshuffleproducer(self, op):
        return self.evaluate(op.input)

    def myriasplitconsumer(self, op):
        return self.evaluate(op.input)

    def myriasplitproducer(self, op):
        return self.evaluate(op.input)

    def myriastore(self, op):
        return self.store(op)

    def myriastoretemp(self, op):
        return self.storetemp(op)

    def myriaappendtemp(self, op):
        return self.appendtemp(op)

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

    def myriaqueryscan(self, op):
        return self.tables.get_sql_output(op.sql).elements()
