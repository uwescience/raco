import unittest

from raco.utility import real_str
from collections import OrderedDict


class TestUtility(unittest.TestCase):
    def test_real_str_int(self):
        self.assertEqual(real_str(1), str(1))
        self.assertEqual(real_str(1, skip_out=True), str(1))

    def test_real_str_string(self):
        self.assertEqual(real_str("abc"), str("abc"))
        self.assertEqual(real_str("abc", skip_out=True), str("abc"))

    def test_real_str_list(self):
        self.assertEqual(real_str([1, 2]), "[1,2]")
        self.assertEqual(real_str([1, 2], skip_out=True), "1,2")

    def test_real_str_dict(self):
        d = OrderedDict([(1, 2), (3, 4)])
        self.assertEqual(real_str(d), "{1:2,3:4}")
        self.assertEqual(real_str(d, skip_out=True), "1:2,3:4")

    def test_real_str_set(self):
        # Python has no built-in ordered set, so we do not know the item order
        self.assertIn(real_str({1, 2}), ["{1,2}", "{2,1}"])
        self.assertIn(real_str({1, 2}, skip_out=True), ["1,2", "2,1"])

    def test_real_str_nested_collections(self):
        self.assertEqual(real_str([[1, 2], {3: 4}, []]),
                         "[[1,2],{3:4},[]]")
        self.assertEqual(real_str([[1, 2], {3: 4}, []], skip_out=True),
                         "[1,2],{3:4},[]")
