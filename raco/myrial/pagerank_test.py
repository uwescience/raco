"""PageRank unit test.

Example data taken from:
http://select.cs.cmu.edu/code/graphlab/doxygen/html/pagerank_example.html
"""

import collections

import raco.scheme as scheme
import raco.myrial.myrial_test as myrial_test
from raco import types


class PageRankTest(myrial_test.MyrialTestCase):

    edge_table = collections.Counter([
        (0, 3),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 4),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4)])

    edge_schema = scheme.Scheme([("src", types.LONG_TYPE),
                                 ("dst", types.LONG_TYPE)])
    edge_key = "public:adhoc:edges"

    vertex_table = collections.Counter([(x,) for x in range(5)])
    vertex_key = "public:adhoc:vertices"
    vertex_schema = scheme.Scheme([("id", types.LONG_TYPE)])

    def setUp(self):
        super(PageRankTest, self).setUp()

        self.db.ingest(PageRankTest.edge_key,
                       PageRankTest.edge_table,
                       PageRankTest.edge_schema)

        self.db.ingest(PageRankTest.vertex_key,
                       PageRankTest.vertex_table,
                       PageRankTest.vertex_schema)

    def __do_test(self, phile):
        with open(phile) as fh:
            query = fh.read()

        result = self.execute_query(query)
        d = dict(result.elements())

        self.assertAlmostEqual(d[0], 0.23576110832410296)
        self.assertAlmostEqual(d[1], 0.16544845649781043)
        self.assertAlmostEqual(d[2], 0.18370688939571236)
        self.assertAlmostEqual(d[3], 0.3016893082129546)
        self.assertAlmostEqual(d[4], 0.11339423756941983)

    def test_pagerank(self):
        self.__do_test('examples/pagerank.myl')

    def verify_undefined(self, var):
        with self.assertRaises(KeyError):
            self.db.get_temp_table(var)

    def test_pagerank_deadcode(self):
        """Test of page rank with numerous dead code statements."""
        self.__do_test('examples/pagerank_dead.myl')

        # Verify that D0, D1 tables are compiled out
        self.verify_undefined("D0")
        self.verify_undefined("D1")
