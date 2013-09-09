
import collections
import unittest

import raco.fakedb
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
import raco.scheme as scheme

class TestQueryFunctions(unittest.TestCase):

    def setUp(self):

        self.db = raco.fakedb.FakeDatabase()

        # Create a fake database with some initial data
        self.emp_table = collections.Counter([
            # id dept_id name salary
            (1, 2, "Bill Howe", 25000),
            (2,1,"Dan Halperin",90000),
            (3,1,"Andrew Whitaker",5000),
            (4,2,"Shumo Chu",5000),
            (5,1,"Victor Almeida",25000),
            (6,3,"Dan Suciu",90000),
            (7,1,"Magdalena Balazinska",25000)])

        self.emp_schema = scheme.Scheme([("id", "int"),
                                         ("dept_id", "int"),
                                         ("name", "string"),
                                         ("salary", int)])

        self.emp_key = "andrew:adhoc:employee"
        self.db.ingest(self.emp_key, self.emp_table, self.emp_schema)

        self.dept_table = collections.Counter([
            (1,"accounting",5),
            (2,"human resources",2),
            (3,"engineering",2),
            (4,"sales",7)])

        self.dept_schema = scheme.Scheme([("id", "int"),
                                          ("name", "string"),
                                          ("manager", "int")])

        self.dept_key = "andrew:adhoc:department"
        self.db.ingest(self.dept_key, self.dept_table, self.dept_schema)

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


    def test_bag_comp_trivial(self):
        query = """
        emp = SCAN(%s);
        bc = [FROM emp EMIT *];
        DUMP bc;
        """ % self.emp_key

        self.__run_test(query, self.emp_table)

if __name__ == '__main__':
    unittest.main()
