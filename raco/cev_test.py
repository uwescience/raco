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
