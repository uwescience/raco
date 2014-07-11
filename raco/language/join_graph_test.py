import unittest
from raco.language.join_graph import JoinGraph


class JoinGraphTest(unittest.TestCase):
    def test_no_cross_product(self):
        jg = JoinGraph(3)
        jg.add_edge(0, 2, 7, 8)
        jg.add_edge(2, 0, 1, 5)
        jg.add_edge(2, 1, 1, 2)

        jo = jg.choose_left_deep_join_order()
        self.assertIn(tuple(jo), {(0,2,1), (2,1,0), (2,0,1), (1,2,0)})
