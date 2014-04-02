
import collections

import raco.scheme as scheme
import raco.myrial.myrial_test as myrial_test


class SigmaClippingTest(myrial_test.MyrialTestCase):
    points = [25.0, 27.2, 23.4, 25.1, 26.3, 24.9, 23.5, 22.7, 108.2,
              26.2, 25.3, 24.7, 25.0, 26.1, 22.8, 2.2, 24.8, 25.05, 25.15]
    points_tuples = [tuple([x]) for x in points]
    points_table = collections.Counter(points_tuples)

    points_schema = scheme.Scheme([('v', 'float')])
    points_key = "public:adhoc:sc_points"

    def setUp(self):
        super(SigmaClippingTest, self).setUp()

        self.db.ingest(SigmaClippingTest.points_key,
                       SigmaClippingTest.points_table,
                       SigmaClippingTest.points_schema)

        # TODO: Better support for empty relations in the language
        self.db.ingest("empty", collections.Counter(),
                       SigmaClippingTest.points_schema)

    def test_v0(self):
        with open('examples/sigma-clipping-v0.myl') as fh:
            query = fh.read()

        points = [x for x in self.points if x < 28 and x > 22]
        expected = collections.Counter([tuple([x]) for x in points])

        self.check_result(query, expected, skip_json=True)

    def test_v2(self):
        with open('examples/sigma-clipping.myl') as fh:
            query = fh.read()

        points = [x for x in self.points if x < 28 and x > 22]
        expected = collections.Counter([tuple([x]) for x in points])

        self.check_result(query, expected, skip_json=True)
