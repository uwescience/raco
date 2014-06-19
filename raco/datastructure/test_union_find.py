from UnionFind import UnionFind
import copy
import unittest


class testUnionFind(unittest.TestCase):

    uf = UnionFind()
    uf.get_or_insert(1)
    uf.get_or_insert(10)
    uf.get_or_insert(2)
    uf.get_or_insert(5)

    def test_insert_or_get(self):
        uf = copy.deepcopy(self.uf)
        self.assertTrue(1 in uf)
        self.assertTrue(10 in uf)
        self.assertTrue(2 in uf)
        self.assertTrue(5 in uf)
        self.assertEqual(uf.get_or_insert(1), 1)
        self.assertEqual(uf.get_or_insert(5), 5)
        self.assertEqual(uf.get_or_insert(10), 10)
        self.assertEqual(uf.get_or_insert(2), 2)
        self.assertEqual(uf.get_or_insert(30), 30)
        self.assertEqual(uf.get_or_insert(52), 52)

    def test_get(self):
        uf = self.uf
        self.assertEqual(uf[1], 1)
        self.assertEqual(uf[10], 10)

    def test_get_error(self):
        with self.assertRaises(Exception):
            self.uf[52]
        with self.assertRaises(Exception):
            self.uf[30]

    def test_union(self):
        uf = copy.deepcopy(self.uf)
        uf.union(1, 10)
        self.assertEqual(uf.get_or_insert(1), uf.get_or_insert(10))
        uf.union(2, 5)
        self.assertEqual(uf.get_or_insert(2), uf.get_or_insert(5))
        uf.union(2, 1)
        self.assertEqual(uf.get_or_insert(10), uf.get_or_insert(5))