import unittest

from raco.compile import optimize_by_rules
import raco.fakedb
from raco.language.logical import OptLogicalAlgebra
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
import raco.scheme as scheme
import raco.types as types
from .flink import compile_to_flink


class FlinkTestCase(unittest.TestCase):
    """A base for testing the compilation of RACO programs to SQL queries"""

    emp_table = [
        # id dept_id name salary
        (1, 2, "Bill Howe", 25000),
        (2, 1, "Dan Halperin", 90000),
        (3, 1, "Andrew Whitaker", 5000),
        (4, 2, "Shumo Chu", 5000),
        (5, 1, "Victor Almeida", 25000),
        (6, 3, "Dan Suciu", 90000),
        (7, 1, "Magdalena Balazinska", 25000)]

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
        self.parser = parser.Parser()
        self.processor = interpreter.StatementProcessor(self.db)

    def compile_query(self, query):
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        p = self.processor.get_logical_plan()
        p = optimize_by_rules(p, OptLogicalAlgebra.opt_rules())
        return compile_to_flink(query, p)

    def test_simple_scan(self):
        query = """
        emp = scan({emp});
        store(emp, OUTPUT);
        """.format(emp=self.emp_key)
        # Just ensure that it compiles
        self.compile_query(query)

    def test_join(self):
        query = """
        emp = scan({emp});
        emp1 = scan({emp});
        j = [from emp, emp1 where emp1.$0 = emp.$1 emit emp.*, emp1.$2];
        store(j, OUTPUT);
        """.format(emp=self.emp_key)
        # Just ensure that it compiles
        self.compile_query(query)

    def test_semi_join(self):
        query = """
        emp = scan({emp});
        emp1 = scan({emp});
        j = [from emp, emp1 where emp1.$0 = emp.$1 emit emp1.*];
        store(j, OUTPUT);
        """.format(emp=self.emp_key)
        # Just ensure that it compiles
        self.compile_query(query)

    def test_filter_condition(self):
        query = """
        emp = scan({emp});
        emp1 = scan({emp});
        j = [from emp, emp1
             where emp1.$2 = "Magdalena Balazinska" and emp1.$0 = emp.$1
             emit emp1.*];
        store(j, OUTPUT);
        """.format(emp=self.emp_key)
        # Just ensure that it compiles
        self.compile_query(query)
