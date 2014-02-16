"""Test of Myrial's control flow graph generation."""

import collections

import raco.myrial.myrial_test as myrial_test
import raco.scheme as scheme
import networkx as nx

def graph_equals(g1, g2):
  """Test for networkx graph equality."""
  return g1.graph == g2.graph and g1.node == g2.node and g1.adj == g2.adj

class CFGTest(myrial_test.MyrialTestCase):
    points_table = collections.Counter()
    points_schema = scheme.Scheme([('id', 'int'), ('x', 'float'),
                                   ('y','float')])
    points_key = "public:adhoc:points"

    def setUp(self):
        super(CFGTest, self).setUp()

        self.db.ingest(CFGTest.points_key,
                       CFGTest.points_table,
                       CFGTest.points_schema)

    def test_cfg(self):
      query = """
      Point = SCAN(public:adhoc:points);

      DO
        Big = [FROM Point WHERE x * y > 100 EMIT *];
        Continue = [FROM Big, Point EMIT COUNT(*) > 0 AS cnt];
      WHILE Continue;

      DUMP(Big);
      """

      statements = self.parser.parse(query)
      self.processor.evaluate(statements)

      expected = nx.DiGraph()
      expected.add_node(0, defs={"Point"}, uses=set())
      expected.add_node(1, defs={"Big"}, uses={"Point"})
      expected.add_node(2, defs={"Continue"}, uses={"Big", "Point"})
      expected.add_node(3, defs=set(), uses={"Continue"})
      expected.add_node(4, defs=set(), uses={"Big"})

      for i in range(4):
          expected.add_edge(i, i + 1)
      expected.add_edge(3, 1)

      actual = self.processor.cfg
      self.assertTrue(graph_equals(actual, expected))
