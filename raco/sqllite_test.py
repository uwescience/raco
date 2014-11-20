import collections
import unittest
import itertools

from raco.dbconn import DBConnection
from raco.fake_data import FakeData

"""Test the raco to sqllite interface."""


class SQLLiteTest(unittest.TestCase, FakeData):

    def setUp(self):
        self.conn1 = DBConnection()
        self.conn2 = DBConnection()

        self.conn1.add_table("emp", FakeData.emp_schema, FakeData.emp_table)
        self.conn1.add_table("dept", FakeData.dept_schema, FakeData.dept_table)
        self.conn2.add_table("num", FakeData.numbers_schema,
                             FakeData.numbers_table)

    def test_empty_relation(self):
        self.conn1.add_table("emp2", FakeData.emp_schema,
                             collections.Counter())
        emp_out = collections.Counter(self.conn1.get_table('emp2'))
        self.assertEquals(emp_out, collections.Counter())

        scheme_out = self.conn1.get_scheme('emp2')
        self.assertEquals(scheme_out, FakeData.emp_schema)

    def test_scan(self):
        emp_out = collections.Counter(self.conn1.get_table('emp'))
        self.assertEquals(emp_out, FakeData.emp_table)

        dept_out = collections.Counter(self.conn1.get_table('dept'))
        self.assertEquals(dept_out, FakeData.dept_table)

        num_out = collections.Counter(self.conn2.get_table('num'))
        self.assertEquals(num_out, FakeData.numbers_table)

    def test_schema_lookup(self):
        self.assertEquals(self.conn1.get_scheme('emp'), FakeData.emp_schema)
        self.assertEquals(self.conn1.get_scheme('dept'), FakeData.dept_schema)
        self.assertEquals(self.conn2.get_scheme('num'),
                          FakeData.numbers_schema)

    def test_num_tuples(self):
        self.assertEquals(self.conn1.num_tuples('emp'),
                          len(FakeData.emp_table))
        self.assertEquals(self.conn1.num_tuples('dept'),
                          len(FakeData.dept_table))
        self.assertEquals(self.conn2.num_tuples('num'),
                          len(FakeData.numbers_table))

    def test_schema_lookup_key_error(self):
        with self.assertRaises(KeyError):
            sc = self.conn2.get_scheme("emp")

    def test_scan_key_error(self):
        with self.assertRaises(KeyError):
            sc = self.conn1.get_table("num")

    def test_delete_table(self):
        sc = self.conn1.get_scheme("emp")
        self.conn1.delete_table("emp")
        with self.assertRaises(KeyError):
            sc = self.conn1.get_scheme("emp")

    def test_append_table(self):
        self.conn1.append_table("emp", FakeData.emp_table)

        it = itertools.chain(iter(FakeData.emp_table),
                             iter(FakeData.emp_table))
        expected = collections.Counter(it)
        actual = collections.Counter(self.conn1.get_table('emp'))
        self.assertEquals(actual, expected)
