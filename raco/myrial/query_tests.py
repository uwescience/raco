
import collections
import math

import raco.algebra
import raco.fakedb
import raco.myrial.interpreter as interpreter
import raco.scheme as scheme
import raco.myrial.groupby
import raco.myrial.myrial_test as myrial_test
import raco.myrial.exceptions


class TestQueryFunctions(myrial_test.MyrialTestCase):

    emp_table = collections.Counter([
        # id dept_id name salary
        (1, 2, "Bill Howe", 25000),
        (2, 1, "Dan Halperin", 90000),
        (3, 1, "Andrew Whitaker", 5000),
        (4, 2, "Shumo Chu", 5000),
        (5, 1, "Victor Almeida", 25000),
        (6, 3, "Dan Suciu", 90000),
        (7, 1, "Magdalena Balazinska", 25000)])

    emp_schema = scheme.Scheme([("id", "int"),
                                ("dept_id", "int"),
                                ("name", "string"),
                                ("salary", "int")])

    emp_key = "public:adhoc:employee"

    dept_table = collections.Counter([
        (1, "accounting", 5),
        (2, "human resources", 2),
        (3, "engineering", 2),
        (4, "sales", 7)])

    dept_schema = scheme.Scheme([("id", "int"),
                                 ("name", "string"),
                                 ("manager", "int")])

    dept_key = "public:adhoc:department"

    numbers_table = collections.Counter([
        (1, 3),
        (2, 5),
        (3, -2),
        (16, -4.3)])

    numbers_schema = scheme.Scheme([("id", "int"),
                                    ("val", "float")])

    numbers_key = "public:adhoc:numbers"

    def setUp(self):
        super(TestQueryFunctions, self).setUp()

        self.db.ingest(TestQueryFunctions.emp_key,
                       TestQueryFunctions.emp_table,
                       TestQueryFunctions.emp_schema)

        self.db.ingest(TestQueryFunctions.dept_key,
                       TestQueryFunctions.dept_table,
                       TestQueryFunctions.dept_schema)

        self.db.ingest(TestQueryFunctions.numbers_key,
                       TestQueryFunctions.numbers_table,
                       TestQueryFunctions.numbers_schema)

    def test_scan_emp(self):
        query = """
        emp = SCAN(%s);
        STORE(emp, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.emp_table)

    def test_scan_dept(self):
        query = """
        dept = SCAN(%s);
        STORE(dept, OUTPUT);
        """ % self.dept_key

        self.check_result(query, self.dept_table)

    def test_bag_comp_emit_star(self):
        query = """
        emp = SCAN(%s);
        bc = [FROM emp EMIT *];
        STORE(bc, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.emp_table)

    def test_bag_comp_emit_table_wildcard(self):
        query = """
        emp = SCAN(%s);
        bc = [FROM emp EMIT emp.*];
        STORE(bc, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.emp_table)

    def test_hybrid_emit_clause(self):
        query = """
        emp = SCAN(%s);
        dept = SCAN(%s);
        x = [FROM dept, emp as X EMIT 5, X.salary * 2 AS k, X.*, *];
        STORE(x, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        expected = [(5, e[3] * 2) + e + d + e for e in self.emp_table
                    for d in self.dept_table]
        self.check_result(query, collections.Counter(expected))

    salary_filter_query = """
    emp = SCAN(%s);
    rich = [FROM emp WHERE %s > 25 * 10 * 10 * (5 + 5) EMIT *];
    STORE(rich, OUTPUT);
    """

    salary_expected_result = collections.Counter(
        [x for x in emp_table.elements() if x[3] > 25000])

    def test_bag_comp_filter_large_salary_by_name(self):
        query = TestQueryFunctions.salary_filter_query % (self.emp_key,
                                                          'salary')
        self.check_result(query, TestQueryFunctions.salary_expected_result)

    def test_bag_comp_filter_large_salary_by_position(self):
        query = TestQueryFunctions.salary_filter_query % (self.emp_key, '$3')
        self.check_result(query, TestQueryFunctions.salary_expected_result)

    def test_bag_comp_filter_empty_result(self):
        query = """
        emp = SCAN(%s);
        poor = [FROM emp WHERE $3 < (5 * 2) EMIT *];
        STORE(poor, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter()
        self.check_result(query, expected)

    def test_bag_comp_filter_column_compare_ge(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE 2 * $1 >= $0 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if 2 * x[1] >= x[0]])
        self.check_result(query, expected)

    def test_bag_comp_filter_column_compare_le(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $1 <= 2 * $0 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[1] <= 2 * x[0]])
        self.check_result(query, expected)

    def test_bag_comp_filter_column_compare_gt(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE 2 * $1 > $0 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if 2 * x[1] > x[0]])
        self.check_result(query, expected)

    def test_bag_comp_filter_column_compare_lt(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $1 < 2 * $0 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[1] < 2 * x[0]])
        self.check_result(query, expected)

    def test_bag_comp_filter_column_compare_eq(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $0 * 2 == $1 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[0] * 2 == x[1]])
        self.check_result(query, expected)

    def test_bag_comp_filter_column_compare_ne(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $0 // $1 != $1 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[0] / x[1] != x[1]])
        self.check_result(query, expected)

    def test_bag_comp_filter_column_compare_ne2(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $0 // $1 <> $1 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[0] / x[1] != x[1]])
        self.check_result(query, expected)

    def test_bag_comp_filter_minus(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $0 + -$1 == $1 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[0] - x[1] == x[1]])
        self.check_result(query, expected)

    def test_bag_comp_filter_and(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE salary == 25000 AND id > dept_id EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000 and
             x[0] > x[1]])
        self.check_result(query, expected)

    def test_bag_comp_filter_or(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $3 > 25 * 1000 OR id > dept_id EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] > 25000 or
             x[0] > x[1]])
        self.check_result(query, expected)

    def test_bag_comp_filter_not(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE not salary > 25000 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if not x[3] > 25000])
        self.check_result(query, expected)

    def test_bag_comp_filter_or_and(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE salary == 25000 OR salary == 5000 AND
        dept_id == 1 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000 or
             (x[3] == 5000 and x[1] == 1)])
        self.check_result(query, expected)

    def test_bag_comp_filter_or_and_not(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE salary == 25000 OR NOT salary == 5000 AND
        dept_id == 1 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000 or not
             x[3] == 5000 and x[1] == 1])
        self.check_result(query, expected)

    def test_bag_comp_emit_columns(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE dept_id == 1 EMIT $2, salary AS salary];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[2], x[3]) for x in self.emp_table.elements() if x[1] == 1])
        self.check_result(query, expected)

    def test_bag_comp_emit_literal(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp EMIT salary, "bugga bugga"];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[3], "bugga bugga") for x in self.emp_table.elements()])

        self.check_result(query, expected)

    def test_bag_comp_emit_with_math(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp EMIT salary + 5000, salary - 5000, salary // 5000,
        salary * 5000];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[3] + 5000, x[3] - 5000, x[3] / 5000, x[3] * 5000)
             for x in self.emp_table.elements()])
        self.check_result(query, expected)

    def test_bag_comp_rename(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp EMIT name, salary * 2 AS double_salary];
        out = [FROM out WHERE double_salary > 10000 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[2], x[3] * 2) for x in self.emp_table.elements() if
             x[3] * 2 > 10000])

        self.check_result(query, expected)

    join_expected = collections.Counter(
        [('Bill Howe', 'human resources'),
         ('Dan Halperin', 'accounting'),
         ('Andrew Whitaker', 'accounting'),
         ('Shumo Chu', 'human resources'),
         ('Victor Almeida', 'accounting'),
         ('Dan Suciu', 'engineering'),
         ('Magdalena Balazinska', 'accounting')])

    def test_explicit_join(self):
        query = """
        emp = SCAN(%s);
        dept = SCAN(%s);
        out = JOIN(emp, dept_id, dept, id);
        out = [FROM out EMIT $2 AS emp_name, $5 AS dept_name];
        STORE(out, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        self.check_result(query, self.join_expected)

    def test_bagcomp_join_via_names(self):
        query = """
        out = [FROM SCAN(%s) E, SCAN(%s) AS D WHERE E.dept_id == D.id
              EMIT E.name AS emp_name, D.name AS dept_name];
        STORE(out, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        self.check_result(query, self.join_expected)

    def test_bagcomp_join_via_pos(self):
        query = """
        E = SCAN(%s);
        D = SCAN(%s);
        out = [FROM E, D WHERE E.$1 == D.$0
              EMIT E.name AS emp_name, D.$1 AS dept_name];
        STORE(out, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        self.check_result(query, self.join_expected)

    def test_join_with_select(self):
        query = """
        out = [FROM SCAN(%s) AS D, SCAN(%s) E
               WHERE E.dept_id == D.id AND E.salary < 6000
               EMIT E.name AS emp_name, D.name AS dept_name];
        STORE(out, OUTPUT);
        """ % (self.dept_key, self.emp_key)

        expected = collections.Counter([('Andrew Whitaker', 'accounting'),
                                        ('Shumo Chu', 'human resources')])
        self.check_result(query, expected)

    def test_sql_join(self):
        """SQL-style select-from-where join"""

        query = """
        E = SCAN(%s);
        D = SCAN(%s);
        out = SELECT E.name, D.name FROM E, D WHERE E.dept_id = D.id;
        STORE(out, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        self.check_result(query, self.join_expected)

    def test_bagcomp_nested_sql(self):
        """Test nesting SQL inside a bag comprehension"""

        query = """
        out = [FROM (SELECT name, salary
                     FROM SCAN(%s) AS X
                     WHERE salary > 5000) AS Y
               WHERE salary < 80000
               EMIT *];
        STORE(out, OUTPUT);
        """ % (self.emp_key,)

        tuples = [(e[2], e[3]) for e in self.emp_table.elements()
                  if e[3] < 80000 and e[3] > 5000]
        expected = collections.Counter(tuples)

        self.check_result(query, expected)

    def test_sql_nested_sql(self):
        """Test nesting SQL inside SQL"""

        query = """
        out = SELECT Y.name, Y.salary
              FROM (SELECT name, salary
                    FROM SCAN(%s) AS X
                    WHERE salary > 5000) AS Y
              WHERE Y.salary < 80000;
        STORE(out, OUTPUT);
        """ % (self.emp_key,)

        tuples = [(e[2], e[3]) for e in self.emp_table.elements()
                  if e[3] < 80000 and e[3] > 5000]
        expected = collections.Counter(tuples)

        self.check_result(query, expected)

    def test_sql_nested_bagcomp(self):
        """Test nesting a bag comprehension inside SQL"""

        query = """
        out = SELECT Y.name, Y.salary FROM
                [FROM SCAN(%s) AS X WHERE salary > 5000 EMIT X.*] AS Y
                WHERE Y.salary < 80000;
        STORE(out, OUTPUT);
        """ % (self.emp_key,)

        tuples = [(e[2], e[3]) for e in self.emp_table.elements()
                  if e[3] < 80000 and e[3] > 5000]
        expected = collections.Counter(tuples)

        self.check_result(query, expected)

    def test_bagcomp_projection(self):
        """Test that column names are preserved across projection."""
        query = """
        E = SCAN(%s);
        F = [FROM E EMIT $2];
        out = [FROM F EMIT name];
        STORE(out, OUTPUT);
        """ % (self.emp_key,)

        tpls = [tuple([x[2]]) for x in self.emp_table]
        expected = collections.Counter(tpls)
        self.check_result(query, expected)

    def test_bagcomp_no_column_name(self):
        """Test that the system handles an omitted output column name."""
        query = """
        E = SCAN(%s);
        F = [FROM E EMIT salary*E.salary];
        out = [FROM F EMIT $0];
        STORE(out, OUTPUT);
        """ % (self.emp_key,)

        tpls = [tuple([x[3] * x[3]]) for x in self.emp_table]
        expected = collections.Counter(tpls)
        self.check_result(query, expected)

    def test_explicit_cross(self):
        query = """
        out = CROSS(SCAN(%s), SCAN(%s));
        STORE(out, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        tuples = [e + d for e in self.emp_table.elements() for
                  d in self.dept_table.elements()]
        expected = collections.Counter(tuples)

        self.check_result(query, expected)

    def test_bagcomp_cross(self):
        query = """
        out = [FROM SCAN(%s) E, SCAN(%s) AS D EMIT *];
        STORE(out, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        tuples = [e + d for e in self.emp_table.elements() for
                  d in self.dept_table.elements()]
        expected = collections.Counter(tuples)

        self.check_result(query, expected)

    def test_distinct(self):
        query = """
        out = DISTINCT([FROM SCAN(%s) AS X EMIT salary]);
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter([(25000,), (5000,), (90000,)])
        self.check_result(query, expected)

    def test_sql_distinct(self):
        query = """
        out = SELECT DISTINCT salary AS salary FROM SCAN(%s) AS X;
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(set([(x[3],) for x in self.emp_table]))
        self.check_result(query, expected)

    def test_sql_repeated(self):
        query = """
        out = SELECT salary AS salary FROM SCAN(%s) AS X;
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter([(x[3],) for x in self.emp_table])
        self.check_result(query, expected)

    def test_limit(self):
        query = """
        out = LIMIT(SCAN(%s), 3);
        STORE(out, OUTPUT);
        """ % self.emp_key

        result = self.execute_query(query)
        self.assertEquals(len(result), 3)

    def test_sql_limit(self):
        query = """
        out = SELECT * FROM SCAN(%s) as X LIMIT 3;
        STORE(out, OUTPUT);
        """ % self.emp_key

        result = self.execute_query(query)
        self.assertEquals(len(result), 3)

    def test_table_literal_scalar_expression(self):
        query = """
        X = [FROM ["Andrew", (50 * (500 + 500)) AS salary] Z EMIT salary];
        STORE(X, OUTPUT);
        """
        expected = collections.Counter([(50000,)])
        self.check_result(query, expected)

    def test_table_literal_unbox(self):
        query = """
        A = [1 AS one, 2 AS two, 3 AS three];
        B = [1 AS one, 2 AS two, 3 AS three];
        C = [*A.two * *B.three];
        STORE(C, OUTPUT);
        """
        expected = collections.Counter([(6,)])
        self.check_result(query, expected)

    def test_unbox_from_where_single(self):
        query = """
        TH = [25 * 1000];
        emp = SCAN(%s);
        out = [FROM emp WHERE $3 > *TH EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] > 25000])
        self.check_result(query, expected)

    def test_unbox_from_where_multi(self):
        query = """
        TWO = [2];
        FOUR = [4];
        EIGHT = [8];

        emp = SCAN(%s);
        out = [FROM emp WHERE *EIGHT == *TWO**FOUR EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.emp_table)

    def test_unbox_from_where_nary_name(self):
        query = """
        CONST = [25 AS twenty_five, 1000 AS thousand];

        emp = SCAN(%s);
        out = [FROM emp WHERE salary == *CONST.twenty_five *
        *CONST.thousand EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000])

        self.check_result(query, expected)

    def test_unbox_from_where_nary_pos(self):
        query = """
        CONST = [25 AS twenty_five, 1000 AS thousand];

        emp = SCAN(%s);
        out = [FROM emp WHERE salary == *CONST.$0 *
        *CONST.$1 EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000])

        self.check_result(query, expected)

    def test_unbox_from_emit_single(self):
        query = """
        THOUSAND = [1000];
        emp = SCAN(%s);
        out = [FROM emp EMIT salary * *THOUSAND AS salary];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[3] * 1000,) for x in self.emp_table.elements()])
        self.check_result(query, expected)

    def test_unbox_kitchen_sink(self):
        query = """
        C1 = [25 AS a, 100 AS b];
        C2 = [50 AS a, 1000 AS b];

        emp = SCAN(%s);
        out = [FROM emp WHERE salary==*C1.a * *C2.b OR $3==*C1.b * *C2
               EMIT dept_id * *C1.b // *C2.a];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[1] * 2,) for x in self.emp_table.elements() if
             x[3] == 5000 or x[3] == 25000])
        self.check_result(query, expected)

    def test_unbox_arbitrary_expression(self):
        query = """
        emp = SCAN(%s);
        dept = SCAN(%s);
        out = [FROM emp WHERE id > *COUNTALL(dept) EMIT emp.id];
        STORE(out, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        expected = collections.Counter(
            [(x[0],) for x in self.emp_table.elements() if
             x[0] > len(self.dept_table)])
        self.check_result(query, expected)

    def test_unbox_inline_table_literal(self):
        query = """
        emp = SCAN(%s);
        dept = SCAN(%s);
        out = [FROM emp WHERE id > *[1,2,3].$2 EMIT emp.id];
        STORE(out, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        expected = collections.Counter(
            [(x[0],) for x in self.emp_table.elements() if
             x[0] > 3])
        self.check_result(query, expected)

    def __aggregate_expected_result(self, apply_func):
        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[1]].append(t[3])

        tuples = [(key, apply_func(values)) for key, values in
                  result_dict.iteritems()]
        return collections.Counter(tuples)

    def test_max(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT dept_id, MAX(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(max))

    def test_min(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT dept_id, MIN(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(min))

    def test_sum(self):
        query = """
        out = [FROM SCAN(%s) as X EMIT dept_id, SUM(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(sum))

    def test_avg(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT dept_id, AVG(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        def avg(it):
            sum = 0
            cnt = 0
            for val in it:
                sum += val
                cnt += 1
            return sum / cnt

        self.check_result(query, self.__aggregate_expected_result(avg))

    def test_stdev(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT STDEV(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        res = self.execute_query(query)
        tp = res.elements().next()
        self.assertAlmostEqual(tp[0], 34001.8006726)

    def test_count(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT dept_id, COUNT(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(len))

    def test_countall(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT dept_id, COUNTALL()];
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(len))

    def test_count_star(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT dept_id, COUNT(*)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(len))

    def test_count_star_sql(self):
        query = """
        out = SELECT dept_id, COUNT(*) FROM SCAN(%s) AS X;
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(len))

    def test_max_reversed(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT MAX(salary) AS max_salary, dept_id];
        STORE(out, OUTPUT);
        """ % self.emp_key

        ex = self.__aggregate_expected_result(max)
        ex = collections.Counter([(y, x) for (x, y) in ex])
        self.check_result(query, ex)

    def test_compound_aggregate(self):
        query = """
        out = [FROM SCAN(%s) AS X
               EMIT (2 * (MAX(salary) - MIN(salary))) AS range,
                    dept_id AS did];
        out = [FROM out EMIT did AS dept_id, range AS rng];
        STORE(out, OUTPUT);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[1]].append(t[3])

        tuples = [(key, 2 * (max(values) - min(values))) for key, values in
                  result_dict.iteritems()]

        expected = collections.Counter(tuples)
        self.check_result(query, expected)

    def test_aggregate_with_unbox(self):
        query = """
        C = [1 AS one, 2 AS two];
        out = [FROM SCAN(%s) AS X
              EMIT MAX(*C.two * salary) - MIN( *C.$1 * salary) AS range,
                   dept_id AS did];
        out = [FROM out EMIT did AS dept_id, range AS rng];
        STORE(out, OUTPUT);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[1]].append(2 * t[3])

        tuples = [(key, (max(values) - min(values))) for key, values in
                  result_dict.iteritems()]

        expected = collections.Counter(tuples)
        self.check_result(query, expected)

    def test_nary_groupby(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT dept_id, salary, COUNT(name)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[(t[1], t[3])].append(t[2])

        tuples = [key + (len(values),)
                  for key, values in result_dict.iteritems()]
        expected = collections.Counter(tuples)
        self.check_result(query, expected)

    def test_empty_groupby(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT MAX(salary), COUNT($0), MIN(dept_id*4)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter([(90000, len(self.emp_table), 4)])
        self.check_result(query, expected)

    def test_compound_groupby(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT id+dept_id, AVG(salary), COUNT(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[0] + t[1]].append(t[3])

        tuples1 = [(key, sum(values), len(values)) for key, values
                   in result_dict.iteritems()]
        tuples2 = [(t[0], t[1] / t[2], t[2]) for t in tuples1]
        expected = collections.Counter(tuples2)

        self.check_result(query, expected)

    def test_impure_aggregate_colref(self):
        """Test of aggregate column that refers to a grouping column"""
        query = """
        out = [FROM SCAN(%s) AS X EMIT
               ( X.dept_id +  (MAX(X.salary) - MIN(X.salary))) AS val,
               X.dept_id AS did];

        out = [FROM out EMIT did AS dept_id, val AS rng];
        STORE(out, OUTPUT);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[1]].append(t[3])

        tuples = [(key, key + (max(values) - min(values))) for key, values in
                  result_dict.iteritems()]

        expected = collections.Counter(tuples)
        self.check_result(query, expected)

    def test_impure_aggregate_unbox(self):
        """Test of an aggregate column that contains an unbox."""
        query = """
        TWO = [2];
        out = [FROM SCAN(%s) AS X
               EMIT (*TWO * (MAX(salary) - MIN(salary))) AS range,
                     dept_id AS did];
        out = [FROM out EMIT did AS dept_id, range AS rng];
        STORE(out, OUTPUT);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[1]].append(t[3])

        tuples = [(key, 2 * (max(values) - min(values))) for key, values in
                  result_dict.iteritems()]

        expected = collections.Counter(tuples)
        self.check_result(query, expected)

    def test_aggregate_illegal_colref(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT
               X.dept_id + COUNT(X.salary) AS val];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(raco.myrial.groupby.InvalidAttributeRefException):  # noqa
            self.check_result(query, None)

    def test_nested_aggregates_are_illegal(self):
        query = """
        out = [FROM SCAN(%s) AS X
               EMIT id+dept_id, MIN(53 + MAX(salary)) AS foo];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(raco.myrial.groupby.NestedAggregateException):
            self.check_result(query, collections.Counter())

    def test_standalone_countall(self):
        query = """
        out = COUNTALL(SCAN(%s));
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter([(len(self.emp_table),)])
        self.check_result(query, expected)

    def test_multiway_bagcomp_with_unbox(self):
        """Return all employees in accounting making less than 30000"""
        query = """
        Salary = [30000];
        Dept = ["accounting"];

        out = [FROM SCAN(%s) AS E, SCAN(%s) AS D
               WHERE E.dept_id == D.id AND D.name == *Dept
               AND E.salary < *Salary EMIT E.$2 AS name];
        STORE(out, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        expected = collections.Counter([
            ("Andrew Whitaker",),
            ("Victor Almeida",),
            ("Magdalena Balazinska",)])
        self.check_result(query, expected)

    def test_duplicate_bagcomp_aliases_are_illegal(self):
        query = """
        X = SCAN(%s);
        out = [FROM X, X EMIT *];
        STORE(out, OUTPUT);
        """ % (self.emp_key,)

        with self.assertRaises(interpreter.DuplicateAliasException):
            self.check_result(query, collections.Counter())

    def test_bagcomp_column_index_out_of_bounds(self):
        query = """
        E = SCAN(%s);
        D = SCAN(%s);
        out = [FROM E, D WHERE E.$1 == D.$77
              EMIT E.name AS emp_name, D.$1 AS dept_name];
        STORE(out, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        with self.assertRaises(raco.myrial.exceptions.ColumnIndexOutOfBounds):
            self.check_result(query, collections.Counter())

    def test_abs(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT id, ABS(val)];
        STORE(out, OUTPUT);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a, abs(b)) for a, b in self.numbers_table.elements()])
        self.check_result(query, expected)

    def test_ceil(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT id, CEIL(val)];
        STORE(out, OUTPUT);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a, math.ceil(b)) for a, b in self.numbers_table.elements()])
        self.check_result(query, expected)

    def test_cos(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT id, COS(val)];
        STORE(out, OUTPUT);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a, math.cos(b)) for a, b in self.numbers_table.elements()])
        self.check_result(query, expected)

    def test_floor(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT id, FLOOR(val)];
        STORE(out, OUTPUT);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a, math.floor(b)) for a, b in self.numbers_table.elements()])
        self.check_result(query, expected)

    def test_log(self):
        query = """
        out = [FROM SCAN(%s) AS X WHERE val > 0 EMIT id, LOG(val)];
        STORE(out, OUTPUT);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a, math.log(b)) for a, b in self.numbers_table.elements()
             if b > 0])
        self.check_result(query, expected)

    def test_sin(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT id, SIN(val)];
        STORE(out, OUTPUT);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a, math.sin(b)) for a, b in self.numbers_table.elements()])
        self.check_result(query, expected)

    def test_sqrt(self):
        query = """
        out = [FROM SCAN(%s) X WHERE val >= 0 EMIT id, SQRT(val)];
        STORE(out, OUTPUT);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a, math.sqrt(b)) for a, b in self.numbers_table.elements()
             if b >= 0])
        self.check_result(query, expected)

    def test_tan(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT id, TAN(val)];
        STORE(out, OUTPUT);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a, math.tan(b)) for a, b in self.numbers_table.elements()])
        self.check_result(query, expected)

    def test_pow(self):
        query = """
        THREE = [3];
        out = [FROM SCAN(%s) X EMIT id, POW(X.val, *THREE)];
        STORE(out, OUTPUT);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a, pow(b, 3)) for a, b in self.numbers_table.elements()])
        self.check_result(query, expected)

    def test_no_such_relation(self):
        query = """
        out = [FROM SCAN(foo:bar:baz) x EMIT id, TAN(val)];
        STORE(out, OUTPUT);
        """

        with self.assertRaises(raco.myrial.interpreter.NoSuchRelationException):  # noqa
            self.check_result(query, collections.Counter())

    def test_scan_error(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT id, !!!FROG(val)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(raco.myrial.exceptions.MyrialCompileException):
            self.check_result(query, collections.Counter())

    def test_parse_error(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT $(val)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(raco.myrial.exceptions.MyrialCompileException):
            self.check_result(query, collections.Counter())

    def test_no_such_udf(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT FooFunction(X.salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(raco.myrial.exceptions.NoSuchFunctionException):
            self.check_result(query, collections.Counter())

    def test_duplicate_udf(self):
        query = """
        DEF foo(x, y): x + y;
        DEF bar(): 7;
        DEF foo(x): -1 * x;

        out = [FROM SCAN(%s) AS X EMIT foo(X.salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(raco.myrial.exceptions.DuplicateFunctionDefinitionException):  # noqa
            self.check_result(query, collections.Counter())

    def test_invalid_argument_udf(self):
        query = """
        DEF Foo(x, y): cos(x) * sin(y);
        out = [FROM SCAN(%s) AS X EMIT Foo(X.salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(raco.myrial.exceptions.InvalidArgumentList):
            self.check_result(query, collections.Counter())

    def test_undefined_variable_udf(self):
        query = """
        DEF Foo(x, y): cos(x) * sin(z);
        out = [FROM SCAN(%s) AS X EMIT Foo(X.salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(raco.myrial.exceptions.UndefinedVariableException):  # noqa
            self.check_result(query, collections.Counter())

    def test_duplicate_variable_udf(self):
        query = """
        DEF Foo(x, x): cos(x) * sin(x);
        out = [FROM SCAN(%s) AS X EMIT Foo(X.salary, X.dept_id)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(raco.myrial.exceptions.DuplicateVariableException):  # noqa
            self.check_result(query, collections.Counter())

    def test_triangle_udf(self):
        query = """
        DEF Triangle(a,b): (a*b)//2;

        out = [FROM SCAN(%s) AS X EMIT id, Triangle(X.salary, dept_id) AS t];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter([(t[0], t[1] * t[3] / 2) for t in self.emp_table])  # noqa
        self.check_result(query, expected)

    def test_noop_udf(self):
        expr = "30 + 15 // 7 + -45"

        query = """
        DEF Noop(): %s;

        out = [Noop() AS t];
        STORE(out, OUTPUT);
        """ % expr

        val = eval(expr)
        expected = collections.Counter([(val,)])
        self.check_result(query, expected)

    def test_composition_udf(self):
        query = """
        DEF Add7(x): x + 7;
        DEF Add6(x): x + 6;
        out = [FROM SCAN(%s) AS X EMIT id, Add6(Add7(Add6(X.salary)))];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter([(t[0], t[3] + 19)
                                        for t in self.emp_table])
        self.check_result(query, expected)

    def test_nested_udf(self):
        query = """
        DEF Add7(x): x + 7;
        DEF Add10(x): Add7(x) + 3;
        out = [FROM SCAN(%s) AS X EMIT id, Add10(X.salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter([(t[0], t[3] + 10)
                                        for t in self.emp_table])
        self.check_result(query, expected)

    def test_regression_150(self):
        """Repeated invocation of a UDF."""

        query = """
        DEF transform(x): pow(10, x/pow(2,16)*3.5);
        out = [FROM SCAN(%s) AS X EMIT id, transform(salary),
               transform(dept_id)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        def tx(x):
            return pow(10, float(x) / pow(2, 16) * 3.5)

        expected = collections.Counter([(t[0], tx(t[3]), tx(t[1]))
                                        for t in self.emp_table])
        self.check_result(query, expected)

    def test_safediv_2_function(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT SafeDiv(X.salary,X.dept_id-1)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(t[3] / (t[1] - 1) if t[1] - 1 > 0 else 0,)
             for t in self.emp_table])
        self.check_result(query, expected)

    def test_safediv_3_function(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT SafeDiv(X.salary,X.dept_id-1,42)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(t[3] / (t[1] - 1) if t[1] - 1 > 0 else 42,)
             for t in self.emp_table])
        self.check_result(query, expected)

    def test_answer_to_everything_function(self):
        query = """
        out = [TheAnswerToLifeTheUniverseAndEverything()];
        STORE(out, OUTPUT);
        """

        expected = collections.Counter([(42,)])
        self.check_result(query, expected)

    def test_running_mean_sapply(self):
        query = """
        APPLY RunningMean(value) {
            [0 AS _count, 0 AS _sum];
            [_count + 1 AS _count, _sum + value AS _sum];
            _sum / _count;
        };
        out = [FROM SCAN(%s) AS X EMIT id, RunningMean(X.salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        tps = []
        _sum = 0
        _count = 0
        for emp in self.emp_table:
            _sum += emp[3]
            _count += 1
            tps.append((emp[0], float(_sum) / _count))

        self.check_result(query, collections.Counter(tps))

    def test_sapply_multi_invocation(self):
        query = """
        APPLY RunningSum(x) {
            [0 AS _sum];
            [_sum + x AS _sum];
            _sum;
        };
        out = [FROM SCAN(%s) AS X
               EMIT id, RunningSum(X.salary), RunningSum(id)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        tps = []
        _sum1 = 0
        _sum2 = 0
        for emp in self.emp_table:
            _sum1 += emp[3]
            _sum2 += emp[0]
            tps.append((emp[0], _sum1, _sum2))

        self.check_result(query, collections.Counter(tps))

    def test_118_regression(self):
        """Regression test for https://github.com/uwescience/datalogcompiler/issues/118"""  # noqa
        query = """
        out = [FROM SCAN(%s) AS X WHERE dept_id = 2 AND salary = 5000 EMIT id];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[0],) for x in self.emp_table.elements()
             if x[1] == 2 and x[3] == 5000])
        self.check_result(query, expected)

    def test_scan_emp_empty_statement(self):
        """Test with an empty statement."""
        query = """
        ;;;
        emp = SCAN(%s);
        STORE(emp, OUTPUT);;;
        """ % self.emp_key

        self.check_result(query, self.emp_table)

    def test_empty_statement_parse(self):
        """Program that contains nothing but empty statements."""
        query = ";"

        statements = self.parser.parse(";")
        self.processor.evaluate(statements)
        plan = self.processor.get_logical_plan()
        self.assertEquals(plan, raco.algebra.Sequence())

    def test_case_binary(self):
        query = """
        emp = SCAN(%s);
        rich = [FROM emp EMIT id, CASE WHEN salary > 15000
                THEN salary // salary
                ELSE 0 // salary END];
        STORE(rich, OUTPUT);
        """ % self.emp_key

        def func(y):
            if y > 15000:
                return 1
            else:
                return 0

        expected = collections.Counter(
            [(x[0], func(x[3])) for x in self.emp_table.elements()])
        self.check_result(query, expected)

    def test_case_ternary(self):
        query = """
        emp = SCAN(%s);
        rich = [FROM emp EMIT id,
                CASE WHEN salary <= 5000 THEN "poor"
                     WHEN salary <= 25000 THEN "middle class"
                     ELSE "rich"
                END];
        STORE(rich, OUTPUT);
        """ % self.emp_key

        def func(y):
            if y <= 5000:
                return 'poor'
            elif y <= 25000:
                return 'middle class'
            else:
                return 'rich'

        expected = collections.Counter(
            [(x[0], func(x[3])) for x in self.emp_table.elements()])
        self.check_result(query, expected)

    def test_case_aggregate(self):
        query = """
        emp = SCAN(%s);
        rich = [FROM emp EMIT SUM(3 * CASE WHEN salary > 15000
                THEN 1 ELSE 0 END)];
        STORE(rich, OUTPUT);
        """ % self.emp_key

        _sum = 3 * len([x for x in self.emp_table.elements()
                        if x[3] > 15000])
        self.check_result(query, collections.Counter([(_sum,)]))

    def test_case_unbox(self):
        query = """
        TH = [15000];
        A = [1 AS one, 2 AS two, 3 AS three];
        emp = SCAN(%s);
        rich = [FROM emp EMIT SUM(*A.three * CASE WHEN salary > *TH
                THEN 1 ELSE 0 END)];
        STORE(rich, OUTPUT);
        """ % self.emp_key

        _sum = 3 * len([x for x in self.emp_table.elements()
                        if x[3] > 15000])
        self.check_result(query, collections.Counter([(_sum,)]))

    def test_default_column_names(self):
        with open('examples/groupby1.myl') as fh:
            query = fh.read()
        self.execute_query(query)
        scheme = self.db.get_scheme('OUTPUT')
        self.assertEquals(scheme.getName(0), "_COLUMN0_")
        self.assertEquals(scheme.getName(1), "id")

    def test_worker_id(self):
        query = """
        X = [FROM SCAN(%s) AS X EMIT X.id, WORKER_ID()];
        STORE(X, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter([(x[0], 0) for x
                                        in self.emp_table.elements()])
        self.check_result(query, expected)
