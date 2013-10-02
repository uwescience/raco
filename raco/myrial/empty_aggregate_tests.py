"""Test of aggregations over empty relations.

Aggregation queries without grouping should return sensible default values:
COUNT(empty) == 0
SUM(empty) == 0
"""

import collections

import raco.scheme as scheme
import raco.myrial.myrial_test as myrial_test

class EmptyAggregateTests(myrial_test.MyrialTestCase):

    empty_schema = scheme.Scheme([("v", "int")])
    empty_key = "andrew:adhoc:empty"

    def setUp(self):
        super(EmptyAggregateTests, self).setUp()

        self.db.ingest(EmptyAggregateTests.empty_key,
                       collections.Counter(),
                       EmptyAggregateTests.empty_schema)


    def test_count(self):
        query = """
        X = [FROM X=SCAN(%s) EMIT COUNT(v)];
        DUMP(X);
        """ % self.empty_key

        self.run_test(query, collections.Counter([(0,)]))

    def test_sum(self):
        query = """
        X = [FROM X=SCAN(%s) EMIT SUM(v)];
        DUMP(X);
        """ % self.empty_key

        self.run_test(query, collections.Counter([(0,)]))
