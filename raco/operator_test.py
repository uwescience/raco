import unittest
import collections

import raco.fakedb
import raco.myrial.myrial_test as myrial_test
import raco.scheme as scheme
from raco.algebra import *
from raco.expression import *
import raco.relation_key as relation_key


class TestQueryFunctions(myrial_test.MyrialTestCase):

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

    emp_key = relation_key.RelationKey.from_string("andrew:adhoc:employee")


class OperatorTest(unittest.TestCase):
    def setUp(self):
        self.db = raco.fakedb.FakeDatabase()
        self.db.ingest(TestQueryFunctions.emp_key,
                       TestQueryFunctions.emp_table,
                       TestQueryFunctions.emp_schema)

    def test_stateful_apply(self):
        scan = Scan(TestQueryFunctions.emp_key, TestQueryFunctions.emp_schema)

        iterex = StateLiteral()
        initex = Literal(0)
        updateex = PLUS(StateLiteral(), NumericLiteral(1))

        sapply = StatefulApply([("index", iterex)], [initex], [updateex], scan)
        result = collections.Counter(self.db.evaluate(sapply))
        self.assertEqual(len(result), len(TestQueryFunctions.emp_table))
        self.assertEqual([x[0] for x in result], range(7))
