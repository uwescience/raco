from collections import Counter
from nose.plugins.skip import SkipTest
import os
import subprocess
import unittest

from raco.compile import optimize_by_rules
import raco.fakedb
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
import raco.scheme as scheme
import raco.types as types
from .flink import compile_to_flink
from .flink_rules import FlinkAlgebra


class FlinkTestCase(unittest.TestCase):
    """A base for testing the compilation of RACO programs to SQL queries"""

    emp_table = Counter([
        # id dept_id name salary
        (1, 2, "Bill Howe", 25000),
        (2, 1, "Dan Halperin", 90000),
        (3, 1, "Andrew Whitaker", 5000),
        (4, 2, "Shumo Chu", 5000),
        (5, 1, "Victor Almeida", 25000),
        (6, 3, "Dan Suciu", 90000),
        (7, 1, "Magdalena Balazinska", 25000)])

    emp_schema = scheme.Scheme([("id", types.INT_TYPE),
                                ("dept_id", types.INT_TYPE),
                                ("name", types.STRING_TYPE),
                                ("salary", types.LONG_TYPE)])

    emp_key = "public:adhoc:employee"

    def setUp(self):
        self.db = raco.fakedb.FakeDatabase()
        self.db.ingest(self.emp_key,
                       self.emp_table,
                       self.emp_schema)
        self.parse_db = raco.fakedb.FakeDatabase()
        self.parse_db.ingest(self.emp_key,
                             self.emp_table,
                             self.emp_schema)
        self.parser = parser.Parser()
        self.processor = interpreter.StatementProcessor(self.db)

    def check_output_and_print_stderr(self, args, java_program):
        """Run the specified command. If it does not exit cleanly, print the
        stderr of the command to stdout. Note that stderr prints are displayed
        as tests run, whereas stdout prints show up next to the failed test. We
        want the latter."""
        try:
            subprocess.check_output(args, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print java_program
            print e.output
            self.fail()

    def compile_query(self, query, output='OUTPUT'):
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        p = self.processor.get_logical_plan()

        # Evaluate the plan given and save the result
        self.parse_db.evaluate(p)
        parser_ans = self.parse_db.get_table(output)

        p = optimize_by_rules(p, FlinkAlgebra.opt_rules())

        # Evaluate the optimized plan and compare the result
        self.db.evaluate(p)
        self.assertEquals(parser_ans, self.db.get_table(output))

        query_str = compile_to_flink(query, p)

        flink_path = os.environ.get('FLINK_PATH')
        if flink_path is not None:
            fname = os.path.join(flink_path, "FlinkQuery.java")
            with open(fname, "w") as outfile:
                outfile.write(query_str)
            cmd = ["javac",
                   "-cp", "{flink}/lib/flink-core-0.6-incubating.jar:{flink}/lib/flink-java-0.6-incubating.jar".format(flink=flink_path),  # noqa
                   fname]
            self.check_output_and_print_stderr(cmd, query_str)
            return query_str
        else:
            raise SkipTest()

    def test_simple_scan(self):
        query = """
        emp = scan({emp});
        store(emp, OUTPUT);
        """.format(emp=self.emp_key)
        self.compile_query(query)

    def test_project(self):
        query = """
        emp = scan({emp});
        e1 = [from emp emit $3, $2];
        store(e1, OUTPUT);
        """.format(emp=self.emp_key)
        self.compile_query(query)

    def test_simple_map(self):
        query = """
        emp = scan({emp});
        e1 = [from emp emit $3, "hi", $2];
        store(e1, OUTPUT);
        """.format(emp=self.emp_key)
        self.compile_query(query)

    def test_join(self):
        query = """
        emp = scan({emp});
        emp1 = scan({emp});
        j = [from emp, emp1 where emp1.$0 = emp.$0 emit emp.*, emp1.$2];
        store(j, OUTPUT);
        """.format(emp=self.emp_key)
        self.compile_query(query)

    def test_join_reorder(self):
        query = """
        emp = scan({emp});
        emp1 = scan({emp});
        j = [from emp, emp1 where emp1.id = emp.id
             emit emp.salary, emp.dept_id, emp1.name];
        store(j, OUTPUT);
        """.format(emp=self.emp_key)
        self.compile_query(query)

    def test_semi_join(self):
        query = """
        emp = scan({emp});
        emp1 = scan({emp});
        j = [from emp, emp1 where emp1.$0 = emp.$0 emit emp1.*];
        store(j, OUTPUT);
        """.format(emp=self.emp_key)
        self.compile_query(query)

    def test_filter_condition(self):
        query = """
        emp = scan({emp});
        emp1 = scan({emp});
        j = [from emp, emp1
             where (emp1.$2 = "Magdalena Balazinska"
                    or emp1.salary < 25000)
               and emp1.$0 = emp.$0
             emit emp1.*];
        store(j, OUTPUT);
        """.format(emp=self.emp_key)
        self.compile_query(query)

    def test_aggregate(self):
        query = """
        emp = scan({emp});
        ans = [from emp
               emit max(salary)];
        store(ans, OUTPUT);
        """.format(emp=self.emp_key)
        self.compile_query(query)

    def test_group_by_agg_long(self):
        query = """
        emp = scan({emp});
        ans = [from emp
               emit dept_id, max(salary)];
        store(ans, OUTPUT);
        """.format(emp=self.emp_key)
        self.compile_query(query)

    def test_group_by_agg_string(self):
        query = """
        emp = scan({emp});
        ans = [from emp
               emit salary, max(name)];
        store(ans, OUTPUT);
        """.format(emp=self.emp_key)
        self.compile_query(query)

    def test_group_by_multi_agg(self):
        query = """
        emp = scan({emp});
        ans = [from emp
               emit max(salary), min(salary)];
        store(ans, OUTPUT);
        """.format(emp=self.emp_key)
        self.compile_query(query)
