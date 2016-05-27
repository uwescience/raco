from collections import Counter
import sqlalchemy
import unittest

import raco.algebra as algebra
from raco.compile import optimize_by_rules
from raco.backends.logical import OptLogicalAlgebra
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
import raco.scheme as scheme
from .catalog import SQLCatalog
import raco.types as types


class SQLTestCase(unittest.TestCase):
    """A base for testing the compilation of RACO programs to SQL queries"""

    emp_table = [
        # id dept_id name salary
        (0, 1, "Hank Levy", 1000000, -1),
        (1, 2, "Bill Howe", 25000, 0),
        (2, 1, "Dan Halperin", 90000, 0),
        (3, 1, "Andrew Whitaker", 5000, 0),
        (4, 2, "Shumo Chu", 5000, 0),
        (5, 1, "Victor Almeida", 25000, 0),
        (6, 3, "Dan Suciu", 90000, 0),
        (7, 1, "Magdalena Balazinska", 25000, 0)]

    emp_schema = scheme.Scheme([("id", types.INT_TYPE),
                                ("dept_id", types.INT_TYPE),
                                ("name", types.STRING_TYPE),
                                ("salary", types.LONG_TYPE),
                                ("mgr_id", types.INT_TYPE)])

    emp_key = "public:adhoc:employee"

    def setUp(self):
        # SQLAlchemy
        self.db = SQLCatalog(sqlalchemy.
                             create_engine('sqlite:///:memory:', echo=True))
        self.db.add_table(self.emp_key, self.emp_schema)
        self.db.add_tuples(self.emp_key, self.emp_schema, self.emp_table)
        # MyriaL
        self.parser = parser.Parser()
        self.processor = interpreter.StatementProcessor(self.db)

    def query_to_phys_plan(self, query, **kwargs):
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        p = self.processor.get_logical_plan(**kwargs)
        p = optimize_by_rules(p, OptLogicalAlgebra.opt_rules())
        if isinstance(p, (algebra.Store, algebra.StoreTemp)):
            p = p.input
        return p

    def execute(self, query, expected, **kwargs):
        p = self.query_to_phys_plan(query, **kwargs)
        ans = self.db.evaluate(p)
        self.assertEquals(expected, Counter(ans))
