
import collections
import unittest

import raco.fakedb
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
import raco.scheme as scheme

class TestQueryFunctions(unittest.TestCase):

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

    emp_key = "andrew:adhoc:employee"

    dept_table = collections.Counter([
        (1,"accounting",5),
        (2,"human resources",2),
        (3,"engineering",2),
        (4,"sales",7)])

    dept_schema = scheme.Scheme([("id", "int"),
                                 ("name", "string"),
                                 ("manager", "int")])

    dept_key = "andrew:adhoc:department"

    def setUp(self):

        self.db = raco.fakedb.FakeDatabase()

        self.db.ingest(TestQueryFunctions.emp_key,
                       TestQueryFunctions.emp_table,
                       TestQueryFunctions.emp_schema)

        self.db.ingest(TestQueryFunctions.dept_key,
                       TestQueryFunctions.dept_table,
                       TestQueryFunctions.dept_schema)

        self.parser = parser.Parser()
        self.processor = interpreter.StatementProcessor(self.db)

    def __run_test(self, query, expected):
        '''Run a test query against the fake database'''
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)

        _var, op = self.processor.output_symbols[0]
        actual = self.db.evaluate_to_bag(op)

        self.assertEquals(actual, expected)

    def test_scan_emp(self):
        query = """
        emp = SCAN(%s);
        DUMP emp;
        """ % self.emp_key

        self.__run_test(query, self.emp_table)

    def test_scan_dept(self):
        query = """
        dept = SCAN(%s);
        DUMP dept;
        """ % self.dept_key

        self.__run_test(query, self.dept_table)


    def test_bag_comp_emit_star(self):
        query = """
        emp = SCAN(%s);
        bc = [FROM emp EMIT *];
        DUMP bc;
        """ % self.emp_key

        self.__run_test(query, self.emp_table)

    salary_filter_query = """
    emp = SCAN(%s);
    rich = [FROM emp WHERE %s > 25 * 10 * 10 * (5 + 5) EMIT *];
    DUMP rich;
    """

    salary_expected_result = collections.Counter(
            [x for x in emp_table.elements() if x[3] > 25000])

    def test_bag_comp_filter_large_salary_by_name(self):
        query =  TestQueryFunctions.salary_filter_query % (self.emp_key,
                                                           'salary')
        self.__run_test(query, TestQueryFunctions.salary_expected_result)

    def test_bag_comp_filter_large_salary_by_position(self):
        query =  TestQueryFunctions.salary_filter_query % (self.emp_key, '$3')
        self.__run_test(query, TestQueryFunctions.salary_expected_result)

    def test_bag_comp_filter_empty_result(self):
        query = """
        emp = SCAN(%s);
        poor = [FROM emp WHERE $3 < (5 * 2) EMIT *];
        DUMP poor;
        """ % self.emp_key

        expected = collections.Counter()
        self.__run_test(query, expected)

    def test_bag_comp_filter_column_compare_ge(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE 2 * $1 >= $0 EMIT *];
        DUMP out;
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if 2 * x[1] >= x[0]])
        self.__run_test(query, expected)

    def test_bag_comp_filter_column_compare_le(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $1 <= 2 * $0 EMIT *];
        DUMP out;
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[1] <= 2 * x[0]])
        self.__run_test(query, expected)

    def test_bag_comp_filter_column_compare_gt(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE 2 * $1 > $0 EMIT *];
        DUMP out;
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if 2 * x[1] > x[0]])
        self.__run_test(query, expected)

    def test_bag_comp_filter_column_compare_lt(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $1 < 2 * $0 EMIT *];
        DUMP out;
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[1] < 2 * x[0]])
        self.__run_test(query, expected)

    def test_bag_comp_filter_column_compare_eq(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $0 * 2 == $1 EMIT *];
        DUMP out;
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[0] * 2 == x[1]])
        self.__run_test(query, expected)

    def test_bag_comp_filter_column_compare_ne(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $0 / $1 != $1 EMIT *];
        DUMP out;
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[0] / x[1] != x[1]])
        self.__run_test(query, expected)

    def test_bag_comp_filter_minus(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $0 + -$1 == $1 EMIT *];
        DUMP out;
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[0] - x[1] ==  x[1]])
        self.__run_test(query, expected)

    def test_bag_comp_filter_and(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE salary == 25000 AND id > dept_id EMIT *];
        DUMP out;
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000 and
             x[0] > x[1]])
        self.__run_test(query, expected)

    def test_bag_comp_filter_or(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $3 > 25 * 1000 OR id > dept_id EMIT *];
        DUMP out;
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] > 25000 or
             x[0] > x[1]])
        self.__run_test(query, expected)

    def test_bag_comp_filter_not(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE not salary > 25000 EMIT *];
        DUMP out;
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if not x[3] > 25000])
        self.__run_test(query, expected)

    def test_bag_comp_filter_or_and(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE salary == 25000 OR salary == 5000 AND
        dept_id == 1 EMIT *];
        DUMP out;
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000 or
             (x[3] == 5000 and x[1] == 1)])
        self.__run_test(query, expected)

    def test_bag_comp_filter_or_and_not(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE salary == 25000 OR NOT salary == 5000 AND
        dept_id == 1 EMIT *];
        DUMP out;
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000 or not
             x[3] == 5000 and x[1] == 1])
        self.__run_test(query, expected)

if __name__ == '__main__':
    unittest.main()
