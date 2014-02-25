"""Test of aggregations over empty relations.

Aggregation queries without grouping should return sensible default values:
COUNT(empty) == 0
SUM(empty) == 0
"""

import collections

import raco.myrial.myrial_test as myrial_test


class EmptyAggregateTests(myrial_test.MyrialTestCase):

    def setUp(self):
        super(EmptyAggregateTests, self).setUp()

    def test_count(self):
        query = """
        W = EMPTY(v:int);
        X = [FROM W EMIT COUNT(v)];
        STORE(X, OUTPUT);
        """

        self.check_result(query, collections.Counter([(0,)]))

    def test_sum(self):
        query = """
        W = EMPTY(v:int);
        X = [FROM W EMIT SUM(v)];
        STORE(X, OUTPUT);
        """

        self.check_result(query, collections.Counter([(0,)]))
