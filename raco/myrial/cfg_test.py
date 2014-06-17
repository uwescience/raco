"""Test of Myrial's control flow graph generation."""

import collections

import raco.myrial.myrial_test as myrial_test
import raco.scheme as scheme
from raco import types
import networkx as nx


class CFGTest(myrial_test.MyrialTestCase):
    points_table = collections.Counter()
    points_schema = scheme.Scheme([('id', types.LONG_TYPE),
                                   ('x', types.DOUBLE_TYPE),
                                   ('y', types.DOUBLE_TYPE)])
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

        STORE(Big, OUTPUT);
        """

        statements = self.parser.parse(query)
        self.processor.evaluate(statements)

        expected = nx.DiGraph()
        expected.add_node(0, def_var="Point", uses=set())
        expected.add_node(1, def_var="Big", uses={"Point"})
        expected.add_node(2, def_var="Continue", uses={"Big", "Point"})
        expected.add_node(3, def_var=None, uses={"Continue"})
        expected.add_node(4, def_var=None, uses={"Big"})

        for i in range(4):
            expected.add_edge(i, i + 1)
        expected.add_edge(3, 1)

        actual = self.processor.cfg.graph

        self.assertEquals(actual.adj, expected.adj)
        self.assertEquals(len(actual), len(expected))

        for n in expected:
            self.assertIn(n, actual)
            self.assertEquals(actual.node[n]['uses'], expected.node[n]['uses'])
            self.assertEquals(actual.node[n]['def_var'],
                              expected.node[n]['def_var'])

        live_in, live_out = self.processor.cfg.compute_liveness()

        self.assertEquals(live_out, {0: {'Point'}, 1: {'Point', 'Big'},
                                     2: {'Continue', 'Big', 'Point'},
                                     3: {'Big', 'Point'}, 4: set()})

        self.assertEquals(live_in, {0: set(), 1: {'Point'},
                                    2: {'Big', 'Point'},
                                    3: {'Big', 'Point', 'Continue'},
                                    4: {'Big'}})

    def test_dead_code_elim(self):
        with open('examples/deadcode.myl') as fh:
            query = fh.read()

        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        self.assertEquals(set(self.processor.cfg.graph.nodes()), set(range(9)))

        self.processor.cfg.dead_code_elimination()
        self.assertEquals(set(self.processor.cfg.graph.nodes()), {2, 6, 7, 8})

    def test_bug_245_dead_loop_elim_do_while(self):
        with open('examples/deadcode2.myl') as fh:
            query = fh.read()

        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        self.assertEquals(set(self.processor.cfg.graph.nodes()), set(range(3)))

        self.processor.cfg.dead_loop_elimination()
        self.processor.cfg.dead_code_elimination()
        self.assertEquals(set(self.processor.cfg.graph.nodes()), set())

    def test_dead_loop_interior(self):
        """Test of a dead loop before the end of the program."""
        query = """
        x = [0 as val, 1 as exp];
        y = x;

        do
            x = [from x emit val+1 as val, 2*exp as exp];
        while [from x emit val < 5];
        store(y, OUTPUT);
        """

        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        self.assertEquals(set(self.processor.cfg.graph.nodes()), set(range(5)))

        self.processor.cfg.dead_loop_elimination()
        self.processor.cfg.dead_code_elimination()

        self.assertEquals(set(self.processor.cfg.graph.nodes()), {0, 1, 4})

    def test_two_dead_loops(self):
        """Test of two unrelated dead loops."""
        query = """
        x = [0 as val, 1 as exp];
        y = x;
        z = y;

        do
            x = [from x emit val+1 as val, 2*exp as exp];
        while [from x emit val < 5];
        do
            z = [from z emit val+1 as val, 2*exp as exp];
        while [from z emit val < 5];
        store(y, OUTPUT);
        """

        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        self.assertEquals(set(self.processor.cfg.graph.nodes()), set(range(8)))

        self.processor.cfg.dead_loop_elimination()
        self.processor.cfg.dead_code_elimination()
        self.assertEquals(set(self.processor.cfg.graph.nodes()), {0, 1, 7})

    def test_two_dead_loops_samevar(self):
        """Test that recursive calls to dead_loop_elimination remove
        repeated dead loops reading/writing the same variable."""
        query = """
        x = [0 as val, 1 as exp];
        y = x;

        do
            x = [from x emit val+1 as val, 2*exp as exp];
        while [from x emit val < 5];

        do
            x = [from x emit val+1 as val, 2*exp as exp];
        while [from x emit val < 5];

        store(y, OUTPUT);
        """

        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        self.assertEquals(set(self.processor.cfg.graph.nodes()), set(range(7)))

        self.processor.cfg.dead_loop_elimination()
        self.processor.cfg.dead_code_elimination()
        self.assertEquals(set(self.processor.cfg.graph.nodes()), {0, 1, 6})

    def test_chaining(self):
        query = """
        A = SCAN(public:adhoc:points);
        B = SCAN(public:adhoc:points);
        C = UNIONALL(A, B);
        D = DISTINCT(C);
        E = SCAN(public:adhoc:points);
        F = DIFF(E, D);
        G = DISTINCT(F);
        STORE(G, OUTPUT);
        """

        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        self.assertEquals(len(self.processor.cfg.graph), 8)

        self.processor.cfg.apply_chaining()
        self.assertEquals(len(self.processor.cfg.graph), 1)

    def test_chaining_variable_reuse(self):
        """Test of chaining with re-used variable names."""
        query = """
        X = SCAN(public:adhoc:points);
        Y = SCAN(public:adhoc:points);
        X = [FROM X, Y WHERE X.x == Y.y EMIT Y.*];
        X = DISTINCT(X);
        STORE(X, OUTPUT);
        """
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        self.assertEquals(len(self.processor.cfg.graph), 5)

        self.processor.cfg.apply_chaining()
        self.assertEquals(self.processor.cfg.graph.nodes(), [4])
        self.assertEquals(len(self.processor.cfg.graph.node[4]['uses']), 0)

    def test_chaining_dead_code_elim(self):
        query = """
        Q = DISTINCT(SCAN(public:adhoc:points));
        DO
            A = SCAN(public:adhoc:points);
            B = SCAN(public:adhoc:points);
            P = DISTINCT(A);
            C = DIFF(A, B);
            Continue = COUNTALL(C);
        WHILE Continue;
        STORE(C, OUTPUT);
        X = SCAN(public:adhoc:points);
        """

        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        self.assertEquals(len(self.processor.cfg.graph), 9)

        self.processor.cfg.dead_code_elimination()
        self.assertEquals(set(self.processor.cfg.graph.nodes()),
                          {1, 2, 4, 5, 6, 7})

        self.processor.cfg.apply_chaining()
        self.assertEquals(set(self.processor.cfg.graph.nodes()), {4, 6, 7})
