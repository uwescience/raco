import unittest

from raco.catalog import FromFileCatalog
from raco.catalog import DEFAULT_CARDINALITY

test_file_path = "raco/catalog_tests"


class TestFromFileCatalog(unittest.TestCase):
    def test_default_cardinality_relation(self):
        cut = FromFileCatalog.load_from_file(
            "{p}/default_cardinality_relation.py".format(p=test_file_path))

        self.assertEqual(cut.get_scheme('B').get_names(),
                         ['x', 'y', 'z'])
        self.assertEqual(cut.get_scheme('A').get_types(),
                         ['DOUBLE_TYPE', 'STRING_TYPE'])

        self.assertEqual(cut.num_tuples('A'), DEFAULT_CARDINALITY)
        self.assertEqual(cut.num_tuples('B'), DEFAULT_CARDINALITY)

    def test_set_cardinality_relation(self):
        cut = FromFileCatalog.load_from_file(
            "{p}/set_cardinality_relation.py".format(p=test_file_path))

        self.assertEqual(cut.get_scheme('C').get_names(),
                         ['a', 'b', 'c'])
        self.assertEqual(cut.num_tuples('B'), DEFAULT_CARDINALITY)
        self.assertEqual(cut.num_tuples('C'), 12)

    def test_missing_relation(self):
        cut = FromFileCatalog.load_from_file(
            "{p}/set_cardinality_relation.py".format(p=test_file_path))

        with self.assertRaises(Exception):
            cut.num_tuples('D')

        with self.assertRaises(Exception):
            cut.get_scheme('D')
