
import collections

import raco.scheme as scheme
import raco.myrial.myrial_test as myrial_test

class ReachableTest(myrial_test.MyrialTestCase):

    edge_table = collections.Counter([
        (1, 2),
        (2, 3),
        (3, 4),
        (3, 5),
        (4, 13),
        (5, 4),
        (1, 9),
        (7, 1),
        (6, 1),
        (10, 11),
        (11, 12),
        (12, 10),
        (10, 1)])

    edge_schema = scheme.Scheme([("src", "int"),
                                 ("dst", "int")])
    edge_key = "andrew:adhoc:edges"

    def setUp(self):
        super(ReachableTest, self).setUp()

        self.db.ingest(ReachableTest.edge_key,
                       ReachableTest.edge_table,
                       ReachableTest.edge_schema)


    def test_reachable(self):
        query = """
        Edge = SCAN(%s);
        Source = [addr=1];
        Reachable = Source;
        Delta = Source;

        DO
        NewlyReachable = DISTINCT([FROM Delta, Edge
        WHERE Delta.addr == Edge.src EMIT addr=Edge.dst]);
        Delta = DIFF(NewlyReachable, Reachable);
        Reachable = UNIONALL(Delta, Reachable);
        WHILE Delta;

        DUMP (Reachable);
        """ % self.edge_key

        expected = collections.Counter([
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (9,),
            (13,),
            ])

        self.run_test(query, expected)
