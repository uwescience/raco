
import collections

import raco.scheme as scheme
import raco.myrial.myrial_test as myrial_test
from raco.myrial.exceptions import *
from raco import types


class SetopTestFunctions(myrial_test.MyrialTestCase):

    emp_table1 = collections.Counter([
        (1, 2, "Bill Howe", 25000),
        (1, 2, "Bill Howe", 25000),
        (2, 1, "Dan Halperin", 90000),
        (3, 1, "Andrew Whitaker", 5000),
        (3, 1, "Andrew Whitaker", 5000),
        (4, 2, "Shumo Chu", 5000),
        (5, 1, "Victor Almeida", 25000),
        (6, 3, "Dan Suciu", 90000),
        (7, 1, "Magdalena Balazinska", 25000)])

    emp_key1 = "andrew:adhoc:employee1"

    emp_table2 = collections.Counter([
        (1, 2, "Bill Howe", 25000),
        (7, 1, "Magdalena Balazinska", 25000),
        (7, 1, "Magdalena Balazinska", 25000),
        (8, 2, "JingJing Wang", 47000)])

    emp_key2 = "andrew:adhoc:employee2"

    emp_schema = scheme.Scheme([("id", types.LONG_TYPE),
                                ("dept_id", types.LONG_TYPE),
                                ("name", types.STRING_TYPE),
                                ("salary", types.LONG_TYPE)])

    def setUp(self):
        super(SetopTestFunctions, self).setUp()

        self.db.ingest(SetopTestFunctions.emp_key1,
                       SetopTestFunctions.emp_table1,
                       SetopTestFunctions.emp_schema)

        self.db.ingest(SetopTestFunctions.emp_key2,
                       SetopTestFunctions.emp_table2,
                       SetopTestFunctions.emp_schema)

    def test_unionall(self):
        query = """
        out = SCAN(%s) + SCAN(%s);
        STORE(out, OUTPUT);
        """ % (self.emp_key1, self.emp_key2)

        expected = self.emp_table1 + self.emp_table2
        self.check_result(query, expected)

    def test_union_schema_mismatch(self):
        query = """
        T1 = [FROM SCAN(%s) AS X EMIT id, dept_id, name, salary, 7 as seven];
        out = UNION(T1, SCAN(%s));
        STORE(out, OUTPUT);
        """ % (self.emp_key1, self.emp_key2)

        with self.assertRaises(SchemaMismatchException):
            self.get_logical_plan(query)

    def test_unionall_inline(self):
        query = """
        out = SCAN(%s) + SCAN(%s);
        STORE(out, OUTPUT);
        """ % (self.emp_key1, self.emp_key2)

        expected = self.emp_table1 + self.emp_table2
        self.check_result(query, expected)

    def test_unionall_inline_ternary(self):
        query = """
        out = SCAN(%s) + [FROM SCAN(%s) AS X EMIT *] + SCAN(%s);
        STORE(out, OUTPUT);
        """ % (self.emp_key1, self.emp_key1, self.emp_key1)

        expected = self.emp_table1 + self.emp_table1 + self.emp_table1
        self.check_result(query, expected)

    def test_diff1(self):
        query = """
        out = DIFF(SCAN(%s), SCAN(%s));
        STORE(out, OUTPUT);
        """ % (self.emp_key1, self.emp_key2)

        expected = collections.Counter(
            set(self.emp_table1).difference(set(self.emp_table2)))
        self.check_result(query, expected)

    def test_diff2(self):
        query = """
        out = DIFF(SCAN(%s), SCAN(%s));
        STORE(out, OUTPUT);
        """ % (self.emp_key2, self.emp_key1)

        expected = collections.Counter(
            set(self.emp_table2).difference(set(self.emp_table1)))
        self.check_result(query, expected)

    def test_diff_schema_mismatch(self):
        query = """
        T1 = [FROM SCAN(%s) AS X EMIT id, dept_id, name];
        out = DIFF(SCAN(%s), T1);
        STORE(out, OUTPUT);
        """ % (self.emp_key1, self.emp_key2)

        with self.assertRaises(SchemaMismatchException):
            self.get_logical_plan(query)

    def test_diff_while_schema_mismatch(self):
        query = """
        Orig = [2 as x];
        T1 = [2 as x];
        do
          Bad = diff(T1, Orig);
          T1 = [3 as x, 3 as y];
        while [from Bad emit count(*) > 0];
        store(T1, OUTPUT);
        """

        with self.assertRaises(SchemaMismatchException):
            # TODO Even if executed, this test does not throw exception
            self.get_logical_plan(query)

    def test_diff_while_schema_mismatch2(self):
        query = """
        Orig = [2 as x];
        T1 = [3 as x];
        do
          Bad = diff(T1, Orig);
          T1 = [3 as x, 3 as y];
        while [from Bad emit count(*) > 0];
        store(T1, OUTPUT);
        """

        with self.assertRaises(SchemaMismatchException):
            # TODO If executed, this test loops infinitely
            self.get_logical_plan(query)

    def test_intersect1(self):
        query = """
        out = INTERSECT(SCAN(%s), SCAN(%s));
        STORE(out, OUTPUT);
        """ % (self.emp_key1, self.emp_key2)

        expected = collections.Counter(
            set(self.emp_table2).intersection(set(self.emp_table1)))
        self.check_result(query, expected, skip_json=True)

    def test_intersect2(self):
        query = """
        out = INTERSECT(SCAN(%s), SCAN(%s));
        STORE(out, OUTPUT);
        """ % (self.emp_key2, self.emp_key1)

        expected = collections.Counter(
            set(self.emp_table1).intersection(set(self.emp_table2)))
        self.check_result(query, expected, skip_json=True)

    def test_intersect_schema_mismatch(self):
        query = """
        T1 = [FROM SCAN(%s) AS X EMIT id, dept_id, name];
        out = INTERSECT(T1, SCAN(%s));
        STORE(out, OUTPUT);
        """ % (self.emp_key1, self.emp_key2)

        with self.assertRaises(SchemaMismatchException):
            self.get_logical_plan(query)
