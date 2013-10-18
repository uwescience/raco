
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
    edge_key = "public:adhoc:edges"

    def setUp(self):
        super(ReachableTest, self).setUp()

        self.db.ingest(ReachableTest.edge_key,
                       ReachableTest.edge_table,
                       ReachableTest.edge_schema)


    def test_reachable(self):
        with open ('examples/reachable.myl') as fh:
            query = fh.read()

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
