import unittest

from raco.catalog import FromFileCatalog
from raco.catalog import DEFAULT_CARDINALITY
import os

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

    def test_schema_to_file(self):
        file = "{p}/test_write_catalog.py".format(p=test_file_path)

        if os.path.isfile(file):
            os.remove(file)

        rel_to_add = 'public:adhoc:test'
        FromFileCatalog.scheme_write_to_file(
            file, rel_to_add,
            "{'columnNames': ['grpID'], 'columnTypes': ['LONG_TYPE']}")

        cut = FromFileCatalog.load_from_file(
            "{p}/test_write_catalog.py".format(p=test_file_path))

        self.assertEqual(cut.get_scheme(rel_to_add).get_names(),
                         ['grpID'])
        self.assertEqual(cut.get_scheme(rel_to_add).get_types(),
                         ['LONG_TYPE'])

    def test_schema_append_to_file(self):
        file = "{p}/test_write_catalog.py".format(p=test_file_path)

        if os.path.isfile(file):
            os.remove(file)

        rel_to_add1 = 'public:adhoc:test1'
        rel_to_add2 = 'public:adhoc:test2'

        FromFileCatalog.scheme_write_to_file(
            file, rel_to_add1,
            "{'columnNames': ['grpID1'], 'columnTypes': ['LONG_TYPE']}")

        FromFileCatalog.scheme_write_to_file(
            file, rel_to_add2,
            "{'columnNames': ['grpID2'], 'columnTypes': ['STRING_TYPE']}")

        cut = FromFileCatalog.load_from_file(
            "{p}/test_write_catalog.py".format(p=test_file_path))

        self.assertEqual(cut.get_scheme(rel_to_add1).get_names(),
                         ['grpID1'])
        self.assertEqual(cut.get_scheme(rel_to_add1).get_types(),
                         ['LONG_TYPE'])
        self.assertEqual(cut.get_scheme(rel_to_add2).get_names(),
                         ['grpID2'])
        self.assertEqual(cut.get_scheme(rel_to_add2).get_types(),
                         ['STRING_TYPE'])

    def test_schema_to_file_no_append(self):
        file = "{p}/test_write_catalog.py".format(p=test_file_path)

        if os.path.isfile(file):
            os.remove(file)

        rel_to_add = 'public:adhoc:test'
        FromFileCatalog.scheme_write_to_file(
            "{p}/test_write_catalog.py".format(p=test_file_path), rel_to_add,
            "{'columnNames': ['grpID'], 'columnTypes': ['LONG_TYPE']}")

        with self.assertRaises(IOError):
            FromFileCatalog.scheme_write_to_file(
                "{p}/test_write_catalog.py".format(p=test_file_path),
                rel_to_add,
                "{'columnNames': ['grpID'], 'columnTypes': ['LONG_TYPE']}",
                append=False)
