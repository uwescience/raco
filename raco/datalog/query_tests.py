import collections

import raco.scheme as scheme
import raco.datalog.datalog_test as datalog_test
from raco import types


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

        self.check_result(query, expected, 'EmpDepts')

    def test_filter(self):
        query = """
        RichGuys(name) :- employee(a, b, name, salary), salary > 25000
        """

        expected = collections.Counter(
            [(x[2],) for x in TestQueryFunctions.emp_table.elements()
             if x[3] > 25000])
        self.check_result(query, expected, 'RichGuys')

    def test_count(self):
        query = """
        OutDegree(src, count(dst)) :- Edge(src, dst)
        """

        counter = collections.Counter()
        for (src, _) in self.edge_table.elements():
            counter[src] += 1

        ex = [(src, cnt) for src, cnt in counter.iteritems()]
        expected = collections.Counter(ex)
        self.check_result(query, expected, 'OutDegree')

    def test_sum_reorder(self):
        query = """
        SalaryByDept(sum(salary), dept_id) :- employee(id, dept_id, name, salary);"""  # noqa
        results = collections.Counter()
        for emp in self.emp_table.elements():
            results[emp[1]] += emp[3]
        expected = collections.Counter([(y, x) for x, y in results.iteritems()])  # noqa
        self.check_result(query, expected, 'SalaryByDept')

    def test_aggregate_no_groups(self):
        query = """
        Total(count(x)) :- Edge(x, y)
        """
        expected = collections.Counter([
            (len(self.edge_table),)])
        self.check_result(query, expected, 'Total')

    def test_multiway_join_chained(self):
        query = """
        OneHop(x) :- Edge(1, x);
        TwoHop(x) :- OneHop(y), Edge(y, x);
        ThreeHop(x) :- TwoHop(y), Edge(y, x)
        """

        expected = collections.Counter([(4,), (5,)])
        self.check_result(query, expected, 'ThreeHop')

    def test_triangles(self):
        # TODO. Right now we have do this separately so that the x<y and y<z
        # conditions are not put in the Join, rather rendered as Selects.
        # Myrialang barfs on theta-joins.
        query = """
        T(x,y,z) :- Edge(x,y), Edge(y,z), Edge(z,x);
        A(x,y,z) :- T(x,y,z), x < y, x < z.
        """

        expected = collections.Counter([(3, 5, 4), (10, 11, 12)])
        self.check_result(query, expected, 'A')

    def test_multiway_join(self):
        query = """
        ThreeHop(z) :- Edge(1, x), Edge(x,y), Edge(y, z);
        """
        expected = collections.Counter([(4,), (5,)])
        self.check_result(query, expected, 'ThreeHop')
