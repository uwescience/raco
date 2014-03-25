import unittest
import physprop


class ColumnEquivalenceClassTest(unittest.TestCase):

    def test_cev(self):
        cev = physprop.ColumnEquivalenceClassSet(12)

        for i in range(12):
            self.assertEquals(cev.get_equivalent_columns(i), {i})

        cev.merge(1, 2)
        cev.merge(4, 3)
        cev.merge(3, 2)

        for i in range(1, 5):
            self.assertEquals(cev.get_equivalent_columns(i), {1, 2, 3, 4})

        for i in range(5, 12):
            self.assertEquals(cev.get_equivalent_columns(i), {i})

    def test_cev_idempotent(self):
        cev = physprop.ColumnEquivalenceClassSet(12)

        cev.merge(2, 1)
        cev.merge(4, 3)
        cev.merge(3, 2)
        cev.merge(2, 1)
        cev.merge(4, 3)
        cev.merge(3, 2)

        for i in range(1, 5):
            self.assertEquals(cev.get_equivalent_columns(i), {1, 2, 3, 4})

        for i in range(5, 12):
            self.assertEquals(cev.get_equivalent_columns(i), {i})

    def test_normalize(self):
        cev = physprop.ColumnEquivalenceClassSet(8)

        cev.merge(2, 1)
        cev.merge(4, 3)
        cev.merge(3, 2)
        cev.merge(6, 7)
        cev.merge(0, 7)

        self.assertEquals(cev.normalize(range(8)), {0, 1, 5})

    def test_merge_set(self):
        cev = physprop.ColumnEquivalenceClassSet(8)

        cev.merge(2, 1)
        cev.merge_set([2, 4, 6])
        cev.merge(6, 5)
        cev.merge(3, 6)

        for i in range(1, 7):
            self.assertEquals(cev.get_equivalent_columns(i), set(range(1, 7)))
        self.assertEquals(cev.get_equivalent_columns(7), {7})
