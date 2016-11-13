import collections

import raco.scheme as scheme
import raco.datalog.datalog_test as datalog_test
from raco import types
from raco.backends.myria import MyriaHyperCubeAlgebra


class TestQueryFunctions(datalog_test.DatalogTestCase):
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

    emp_key = "employee"

    dept_table = collections.Counter([
        (1, "accounting", 5),
        (2, "human resources", 2),
        (3, "engineering", 2),
        (4, "sales", 7)])

    dept_schema = scheme.Scheme([("id", types.LONG_TYPE),
                                 ("name", types.STRING_TYPE),
                                 ("manager", types.LONG_TYPE)])

    dept_key = "department"

    edge_table = collections.Counter([
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 3),
        (3, 5),
        (4, 13),
        (5, 4),
        (1, 9),
        (7, 1),
        (6, 1),
        (10, 11),
        (11, 12),
        (12, 10),
        (13, 4),
        (10, 1)])

    edge_schema = scheme.Scheme([("src", types.LONG_TYPE),
                                 ("dst", types.LONG_TYPE)])
    edge_key = "Edge"

    def setUp(self):
        super(TestQueryFunctions, self).setUp()

        self.db.ingest(TestQueryFunctions.emp_key,
                       TestQueryFunctions.emp_table,
                       TestQueryFunctions.emp_schema)

        self.db.ingest(TestQueryFunctions.dept_key,
                       TestQueryFunctions.dept_table,
                       TestQueryFunctions.dept_schema)

        self.db.ingest(TestQueryFunctions.edge_key,
                       TestQueryFunctions.edge_table,
                       TestQueryFunctions.edge_schema)

    def test_simple_join(self):
        expected = collections.Counter(
            [(e[2], d[1]) for e in self.emp_table.elements()
             for d in self.dept_table.elements() if e[1] == d[0]])

        query = """
        EmpDepts(emp_name, dept_name) :- employee(a, dept_id, emp_name, b),
                department(dept_id, dept_name, c)
        """

        self.check_result(query, expected, output='EmpDepts')

    def test_filter(self):
        query = """
        RichGuys(name) :- employee(a, b, name, salary), salary > 25000
        """

        expected = collections.Counter(
            [(x[2],) for x in TestQueryFunctions.emp_table.elements()
             if x[3] > 25000])
        self.check_result(query, expected, output='RichGuys')

    def test_count(self):
        query = """
        OutDegree(src, count(dst)) :- Edge(src, dst)
        """

        counter = collections.Counter()
        for (src, _) in self.edge_table.elements():
            counter[src] += 1

        ex = [(src, cnt) for src, cnt in counter.iteritems()]
        expected = collections.Counter(ex)
        self.check_result(query, expected, output='OutDegree')

    def test_sum_reorder(self):
        query = """
        SalaryByDept(sum(salary), dept_id) :- employee(id, dept_id, name, salary);"""  # noqa
        results = collections.Counter()
        for emp in self.emp_table.elements():
            results[emp[1]] += emp[3]
        expected = collections.Counter([(y, x) for x, y in results.iteritems()])  # noqa
        self.check_result(query, expected, output='SalaryByDept')

    def test_aggregate_no_groups(self):
        query = """
        Total(count(x)) :- Edge(x, y)
        """
        expected = collections.Counter([
            (len(self.edge_table),)])
        self.check_result(query, expected, output='Total')

    def test_multiway_join_chained(self):
        query = """
        OneHop(x) :- Edge(1, x);
        TwoHop(x) :- OneHop(y), Edge(y, x);
        ThreeHop(x) :- TwoHop(y), Edge(y, x)
        """

        expected = collections.Counter([(4,), (5,)])
        self.check_result(query, expected, output='ThreeHop')

    def test_triangles(self):
        # TODO. Right now we have do this separately so that the x<y and y<z
        # conditions are not put in the Join, rather rendered as Selects.
        # Myrialang barfs on theta-joins.
        query = """
        T(x,y,z) :- Edge(x,y), Edge(y,z), Edge(z,x);
        A(x,y,z) :- T(x,y,z), x < y, x < z.
        """

        expected = collections.Counter([(3, 5, 4), (10, 11, 12)])
        self.check_result(query, expected, output='A')

    def test_multiway_join(self):
        query = """
        ThreeHop(z) :- Edge(1, x), Edge(x,y), Edge(y, z);
        """
        expected = collections.Counter([(4,), (5,)])
        self.check_result(query, expected, output='ThreeHop')

    def test_multiway_join_hyper_cube(self):
        query = """
        ThreeHop(z) :- Edge(1, x), Edge(x,y), Edge(y, z);
        """
        expected = collections.Counter([(4,), (5,)])
        self.check_result(query, expected, output='ThreeHop',
                          algebra=MyriaHyperCubeAlgebra)

    def test_union(self):
        query = """
        OUTPUT(b) :- {emp}(a, b, c, d)
        OUTPUT(b) :- {edge}(b, a)
        """.format(emp=self.emp_key, edge=self.edge_key)
        expected = collections.Counter(
            [(b,) for (a, b, c, d) in self.emp_table] +
            [(b,) for (b, a) in self.edge_table]
        )
        self.check_result(query, expected, test_logical=True)

    def test_filter_expression(self):
        query = """
        OUTPUT(a, b, c) :- {emp}(a, b, c, d), d >= 25000, d < 91000
        """.format(emp=self.emp_key)
        expected = collections.Counter(
            [(a, b, c)
             for (a, b, c, d) in self.emp_table
             if (d >= 25000 and d < 91000)]
        )
        self.check_result(query, expected)

    def test_attributes_forward(self):
        """test that attributes are correct amid multiple conditions"""
        query = """
        OUTPUT(a) :- {edge}(a, b), {emp}(c, a, x, y), b=c
        """.format(emp=self.emp_key, edge=self.edge_key)
        expected = collections.Counter(
            [(a,)
             for (a, b) in self.edge_table
             for (c, a2, x, y) in self.emp_table
             if (a == a2 and b == c)]
        )
        self.check_result(query, expected)

    def test_attributes_reverse(self):
        """test that attributes are correct amid multiple conditions and when
        the order of variables in the terms is the opposite of the explicit
        condition"""
        query = """
        OUTPUT(a) :- {edge}(a, b), {emp}(c, a, x, y), c=b
        """.format(emp=self.emp_key, edge=self.edge_key)
        expected = collections.Counter(
            [(a,)
             for (a, b) in self.edge_table
             for (c, a2, x, y) in self.emp_table
             if (a == a2 and b == c)]
        )
        self.check_result(query, expected)

    def test_apply_head(self):
        query = """
        OUTPUT(a/b) :- {emp}(a, b, c, d)
        """.format(emp=self.emp_key)
        expected = collections.Counter([
            (a * 1.0 / b,) for (a, b, _, _) in self.emp_table
        ])
        self.check_result(query, expected)

    def test_aggregate_head(self):
        query = """
        OUTPUT(SUM(a)) :- {emp}(a, b, c, d)
        """.format(emp=self.emp_key)
        expected = collections.Counter([
            (sum(a for (a, _, _, _) in self.emp_table),)
        ])
        self.check_result(query, expected)

    def test_twoaggregate_head(self):
        query = """
        OUTPUT(SUM(a), COUNT(b)) :- {emp}(a, b, c, d)
        """.format(emp=self.emp_key)
        expected = collections.Counter([
            (sum(a for (a, _, _, _) in self.emp_table),
             sum(1 for (_, b, _, _) in self.emp_table))
        ])
        self.check_result(query, expected)

    def test_aggregate_head_group_self(self):
        query = """
        OUTPUT(SUM(a), b) :- {emp}(a, b, c, d)
        """.format(emp=self.emp_key)
        B = set(b for (_, b, _, _) in self.emp_table)
        expected = collections.Counter([
            (sum(a for (a, b, _, _) in self.emp_table
                 if b == b2), b2)
            for b2 in B
        ])
        self.check_result(query, expected)

    def test_aggregate_head_group_swap(self):
        query = """
        OUTPUT(b,SUM(a)) :- {emp}(a, b, c, d)
        """.format(emp=self.emp_key)
        B = set(b for (_, b, _, _) in self.emp_table)
        expected = collections.Counter([
            (b2, sum(a for (a, b, _, _) in self.emp_table
                     if b == b2))
            for b2 in B
        ])
        self.check_result(query, expected)

    def test_binop_aggregates(self):
        query = """
        OUTPUT(SUM(b)+SUM(a)) :- {emp}(a, b, c, d)
        """.format(emp=self.emp_key)
        expected = collections.Counter([
            (sum(b for (a, b, _, _) in self.emp_table) +
             sum(a for (a, b, _, _) in self.emp_table),)
        ])
        self.check_result(query, expected)

    def test_aggregate_of_binop(self):
        query = """
        OUTPUT(SUM(b+a)) :- {emp}(a, b, c, d)
        """.format(emp=self.emp_key)
        expected = collections.Counter(
            [(sum([(a + b) for (a, b, c, d) in self.emp_table]),)])
        self.check_result(query, expected)

    def test_literal_expr(self):
        query = """
        OUTPUT(z+1) :- Edge(z, y)
        """
        expected = collections.Counter([(z + 1,)
                                        for (z, _) in self.edge_table])
        self.check_result(query, expected)
