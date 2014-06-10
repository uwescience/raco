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
        (1, 2, "Bill Howe", 25000),
        (2, 1, "Dan Halperin", 90000),
        (3, 1, "Andrew Whitaker", 5000),
        (4, 2, "Shumo Chu", 5000),
        (5, 1, "Victor Almeida", 25000),
        (6, 3, "Dan Suciu", 90000),
        (7, 1, "Magdalena Balazinska", 25000)])

    emp_schema = scheme.Scheme([("id", "LONG_TYPE"),
                                ("dept_id", "LONG_TYPE"),
                                ("name", "STRING_TYPE"),
                                ("salary", "LONG_TYPE")])

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

        initex = NumericLiteral(-1)
        iterex = NamedStateAttributeRef("count")
        updateex = PLUS(NamedStateAttributeRef("count"),
                        NumericLiteral(1))

        sapply = StatefulApply([("count", iterex)],
                               [("count", initex, updateex)], scan)
        result = collections.Counter(self.db.evaluate(sapply))
        self.assertEqual(len(result), len(TestQueryFunctions.emp_table))
        self.assertEqual([x[0] for x in result], range(7))

    def test_running_mean_stateful_apply(self):
        """Calculate the mean using stateful apply"""
        scan = Scan(TestQueryFunctions.emp_key, TestQueryFunctions.emp_schema)

        initex0 = NumericLiteral(0)
        updateex0 = PLUS(NamedStateAttributeRef("count"),
                         NumericLiteral(1))

        initex1 = NumericLiteral(0)
        updateex1 = PLUS(NamedStateAttributeRef("sum"),
                         NamedAttributeRef("salary"))

        avgex = IDIVIDE(NamedStateAttributeRef("sum"),
                        NamedStateAttributeRef("count"))

        sapply = StatefulApply([("avg", avgex)],
                               [("count", initex0, updateex0),
                                ("sum", initex1, updateex1)], scan)
        result = list(self.db.evaluate(sapply))

        self.assertEqual(len(result), len(TestQueryFunctions.emp_table))
        self.assertEqual([x[0] for x in result][-1], 37857)

        # test whether we can generate json without errors
        from myrialang import compile_to_json, MyriaAlgebra
        from compile import optimize
        import json
        json_string = json.dumps(compile_to_json("", None, optimize([('root', sapply)], LogicalAlgebra, MyriaAlgebra)))  # noqa
        assert json_string

    def test_cast_to_float(self):
        scan = Scan(TestQueryFunctions.emp_key, TestQueryFunctions.emp_schema)
        cast = FLOAT_CAST(NamedAttributeRef("salary"))
        applyop = Apply([("salaryf", cast)], scan)
        res = list(self.db.evaluate(applyop))
        for x in res:
            assert isinstance(x[0], float)
        self.assertEqual([x[0] for x in res],
                         [x[3] for x in TestQueryFunctions.emp_table])
