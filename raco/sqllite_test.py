import collections
import unittest

from raco.sqllite import SQLLiteConnection
from raco.fake_data import FakeData


class SQLLiteTest(unittest.TestCase, FakeData):

    def setUp(self):
        self.conn1 = SQLLiteConnection()
        self.conn2 = SQLLiteConnection()

        self.conn1.add_table("emp", FakeData.emp_schema, FakeData.emp_table)
        self.conn1.add_table("dept", FakeData.dept_schema, FakeData.dept_table)
        self.conn2.add_table("num", FakeData.numbers_schema,
                             FakeData.numbers_table)

    def test_scan(self):
        emp_out = collections.Counter(self.conn1.get_table('emp'))
        self.assertEquals(emp_out, FakeData.emp_table)

        dept_out = collections.Counter(self.conn1.get_table('dept'))
        self.assertEquals(dept_out, FakeData.dept_table)

        num_out = collections.Counter(self.conn2.get_table('num'))
        self.assertEquals(num_out, FakeData.numbers_table)
