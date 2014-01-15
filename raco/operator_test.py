import unittest
import collections

import raco.fakedb
import raco.scheme as scheme
from raco.algebra import *
from raco.expression import *
import raco.relation_key as relation_key


class TestQueryFunctions():

    emp_table = collections.Counter([
        # id dept_id name salary
        (1,2, "Bill Howe", 25000),
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

    def test_counter_stateful_apply(self):
        """Test stateful apply operator that produces a counter"""
        scan = Scan(TestQueryFunctions.emp_key, TestQueryFunctions.emp_schema)

        iterex = StateLiteral()
        initex = NumericLiteral(0)
        updateex = PLUS(StateLiteral(), NumericLiteral(1))

        sapply = StatefulApply([("count", iterex)], [initex], [updateex], scan)
        result = collections.Counter(self.db.evaluate(sapply))
        self.assertEqual(len(result), len(TestQueryFunctions.emp_table))
        self.assertEqual([x[0] for x in result], range(7))

    def test_running_mean_stateful_apply(self):
        """Calculate the mean using two stateful applies.
        One for the sum and another one for the count"""
        scan = Scan(TestQueryFunctions.emp_key, TestQueryFunctions.emp_schema)

        initex0 = NumericLiteral(1)
        iterex0 = StateLiteral()
        updateex0 = PLUS(StateLiteral(), NumericLiteral(1))

        initex1 = NumericLiteral(0)
        iterex1 = PLUS(StateLiteral(), NamedAttributeRef("salary"))
        updateex1 = PLUS(StateLiteral(), NamedAttributeRef("salary"))

        avgex = DIVIDE(NamedAttributeRef("sum"), NamedAttributeRef("count"))

        sapply = StatefulApply([("count", iterex0), ("sum", iterex1)],
                               [initex0, initex1],
                               [updateex0, updateex1], scan)
        avg = Apply([("avg", avgex), ("sum", NamedAttributeRef("sum")),
                     ("count", NamedAttributeRef("count"))], sapply)
        result = list(self.db.evaluate(avg))

        self.assertEqual(len(result), len(TestQueryFunctions.emp_table))
        for x in result:
            self.assertEqual(x[0], x[1]/x[2])
        self.assertEqual([x[2] for x in result], range(1, 8))
        self.assertEqual([x[0] for x in result],
                         [5000, 15000, 18333, 20000, 34000, 43333, 37857])
