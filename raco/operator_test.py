import unittest

import raco.fakedb
from raco.relation_key import RelationKey
from raco.algebra import *
from raco.expression import *
import raco.relation_key as relation_key
from raco.expression import StateVar


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

    emp_schema = scheme.Scheme([("id", types.LONG_TYPE),
                                ("dept_id", types.LONG_TYPE),
                                ("name", types.STRING_TYPE),
                                ("salary", types.LONG_TYPE)])

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
        updateex = PLUS(UnnamedStateAttributeRef(0),
                        NumericLiteral(1))

        sapply = StatefulApply([("count", iterex)],
                               [StateVar("count", initex, updateex)], scan)
        result = collections.Counter(self.db.evaluate(sapply))
        self.assertEqual(len(result), len(TestQueryFunctions.emp_table))
        self.assertEqual([x[0] for x in result], range(7))

    def test_times_equal_uda(self):
        input_op = Scan(TestQueryFunctions.emp_key,
                        TestQueryFunctions.emp_schema)

        init_ex = NumericLiteral(1)
        update_ex = TIMES(NamedStateAttributeRef("value"),
                          NamedAttributeRef("salary"))
        emit_ex = UdaAggregateExpression(NamedStateAttributeRef("value"))

        statemods = [StateVar("value", init_ex, update_ex)]
        gb = GroupBy([UnnamedAttributeRef(1)], [emit_ex], input_op, statemods)
        result = self.db.evaluate_to_bag(gb)

        d = collections.defaultdict(lambda: 1)
        for tpl in TestQueryFunctions.emp_table:
            d[tpl[1]] *= tpl[3]
        expected = collections.Counter(
            [(key, val) for key, val in d.iteritems()])

        self.assertEquals(result, expected)

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
                               [StateVar("count", initex0, updateex0),
                                StateVar("sum", initex1, updateex1)], scan)

        store = Store(RelationKey("OUTPUT"), sapply)
        result = list(self.db.evaluate(sapply))

        self.assertEqual(len(result), len(TestQueryFunctions.emp_table))
        self.assertEqual([x[0] for x in result][-1], 37857)

        # test whether we can generate json without errors
        from raco.backends.myria import (compile_to_json,
                                         MyriaLeftDeepTreeAlgebra)
        from compile import optimize
        import json
        json_string = json.dumps(compile_to_json("", None, optimize(store, MyriaLeftDeepTreeAlgebra())))  # noqa
        assert json_string

    def test_cast_to_float(self):
        scan = Scan(TestQueryFunctions.emp_key, TestQueryFunctions.emp_schema)
        cast = CAST(types.DOUBLE_TYPE, NamedAttributeRef("salary"))
        applyop = Apply([("salaryf", cast)], scan)
        res = list(self.db.evaluate(applyop))
        for x in res:
            assert isinstance(x[0], float)
        self.assertEqual([x[0] for x in res],
                         [x[3] for x in TestQueryFunctions.emp_table])

    def test_projecting_join_scheme(self):
        emp = Scan(TestQueryFunctions.emp_key, TestQueryFunctions.emp_schema)
        emp1 = Scan(TestQueryFunctions.emp_key, TestQueryFunctions.emp_schema)
        pj = ProjectingJoin(condition=BooleanLiteral(True),
                            left=emp, right=emp1)
        names = ([n for n in emp.scheme().get_names()]
                 + ["{n}1".format(n=n) for n in emp.scheme().get_names()])
        self.assertEquals(names, pj.scheme().get_names())

    def test_projecting_join_scheme_no_dups_alternate(self):
        emp = Scan(TestQueryFunctions.emp_key, TestQueryFunctions.emp_schema)
        emp1 = Scan(TestQueryFunctions.emp_key, TestQueryFunctions.emp_schema)
        num_cols = len(emp.scheme())
        # alternate which copy of emp we keep a col from
        refs = [UnnamedAttributeRef(i + (i % 2) * num_cols)
                for i in range(num_cols)]
        pj = ProjectingJoin(condition=BooleanLiteral(True),
                            left=emp, right=emp1, output_columns=refs)
        self.assertEquals(emp.scheme().get_names(), pj.scheme().get_names())

    def test_projecting_join_scheme_no_dups_only_keep_right(self):
        emp = Scan(TestQueryFunctions.emp_key, TestQueryFunctions.emp_schema)
        emp1 = Scan(TestQueryFunctions.emp_key, TestQueryFunctions.emp_schema)
        num_cols = len(emp.scheme())
        # keep only the right child's columns
        refs = [UnnamedAttributeRef(i + num_cols)
                for i in range(num_cols)]
        pj = ProjectingJoin(condition=BooleanLiteral(True),
                            left=emp, right=emp1, output_columns=refs)
        self.assertEquals(emp.scheme().get_names(), pj.scheme().get_names())
