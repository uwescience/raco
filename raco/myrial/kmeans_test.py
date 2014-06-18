"""Unit test of kmeans.

TODO: implement a clustering algorithm that is less sensitive to the
initial cluster selection.  We can't verify the output because this algorithm
chooses initial clusters in a non-robust way.
"""

import collections

import raco.scheme as scheme
import raco.myrial.myrial_test as myrial_test
from raco import types


class KmeansTest(myrial_test.MyrialTestCase):
    points = [(1, 1.0, 1.0), (2, .99, .99), (3, 1.01, 1.01), (4, 10.0, 10.0),
              (5, 10.99, 10.99), (6, 10.01, 10.01), (7, 100.0, 100.0),
              (8, 100.99, 100.99), (9, 100.01, 100.01)]
    points_table = collections.Counter(points)

    points_schema = scheme.Scheme([('id', types.LONG_TYPE),
                                   ('x', types.DOUBLE_TYPE),
                                   ('y', types.DOUBLE_TYPE)])
    points_key = "public:adhoc:points"

    def setUp(self):
        super(KmeansTest, self).setUp()

        self.db.ingest(KmeansTest.points_key,
                       KmeansTest.points_table,
                       KmeansTest.points_schema)

    def test_kmeans(self):
        with open('examples/kmeans.myl') as fh:
            query = fh.read()
        self.execute_query(query, skip_json=True)
