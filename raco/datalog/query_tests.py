import collections
import math
import unittest

import raco.fakedb
import raco.scheme as scheme
import raco.datalog.datalog_test as datalog_test

class TestQueryFunctions(datalog_test.DatalogTestCase):
    emp_table = collections.Counter([
        # id dept_id name salary
        (1, 2, "Bill Howe", 25000),
        (2,1,"Dan Halperin",90000),
        (3,1,"Andrew Whitaker",5000),
        (4,2,"Shumo Chu",5000),
        (5,1,"Victor Almeida",25000),
        (6,3,"Dan Suciu",90000),
        (7,1,"Magdalena Balazinska",25000)])

    emp_schema = scheme.Scheme([("id", "int"),
                                ("dept_id", "int"),
                                ("name", "string"),
                                ("salary", "int")])

    emp_key = "public:adhoc:employee"

    dept_table = collections.Counter([
        (1,"accounting",5),
        (2,"human resources",2),
        (3,"engineering",2),
        (4,"sales",7)])

    dept_schema = scheme.Scheme([("id", "int"),
                                 ("name", "string"),
                                 ("manager", "int")])

    dept_key = "public:adhoc:department"

    def setUp(self):
        super(TestQueryFunctions, self).setUp()

        self.db.ingest(TestQueryFunctions.emp_key,
                       TestQueryFunctions.emp_table,
                       TestQueryFunctions.emp_schema)

        self.db.ingest(TestQueryFunctions.dept_key,
                       TestQueryFunctions.dept_table,
                       TestQueryFunctions.dept_schema)

    def test_simple_join(self):
        expected = collections.Counter(
            [ (e[2], d[1]) for e in self.emp_table.elements()
             for d in self.dept_table.elements() if e[1] == d[0]])

        query = """
        EmpDepts(emp_name, dept_name) :- employee(a, dept_id, emp_name, b),
                department(dept_id, dept_name, c)
        """

        self.run_test(query, expected)
