
import collections

import raco.scheme as scheme
import raco.myrial.myrial_test as myrial_test
from raco import types


class SigmaClippingTest(myrial_test.MyrialTestCase):
    points = [25.0, 27.2, 23.4, 25.1, 26.3, 24.9, 23.5, 22.7, 108.2,
              26.2, 25.3, 24.7, 25.01, 26.1, 22.8, 2.2, 24.8, 25.05, 25.15]
    points_tuples = [(i, x) for i, x in enumerate(points)]
    points_table = collections.Counter(points_tuples)

    points_schema = scheme.Scheme([('id', types.LONG_TYPE), ('v', types.DOUBLE_TYPE)])  # noqa
    points_key = "public:adhoc:sc_points"

    def setUp(self):
        super(SigmaClippingTest, self).setUp()

        self.db.ingest(SigmaClippingTest.points_key,
                       SigmaClippingTest.points_table,
                       SigmaClippingTest.points_schema)

        # TODO: Better support for empty relations in the language
        self.db.ingest("empty", collections.Counter(),
                       SigmaClippingTest.points_schema)

    def run_it(self, query):
        points = [(i, x) for i, x in self.points_tuples if x < 28 and x > 22]
        expected = collections.Counter(points)
        self.check_result(query, expected, output='sc_points_clipped')

    def test_v0(self):
        with open('examples/sigma-clipping-v0.myl') as fh:
            query = fh.read()
        self.run_it(query)

    def test_v2(self):
        with open('examples/sigma-clipping.myl') as fh:
            query = fh.read()
        self.run_it(query)
