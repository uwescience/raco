# -*- coding: UTF-8 -*-

import collections
import math
import md5
from nose.tools import nottest

import raco.algebra
import raco.fakedb
import raco.myrial.interpreter as interpreter
import raco.scheme as scheme
import raco.myrial.groupby
import raco.myrial.myrial_test as myrial_test
from raco import types

from raco.myrial.exceptions import *
from raco.expression import NestedAggregateException
from raco.fake_data import FakeData
from raco.types import LONG_TYPE


class TestQueryFunctions(myrial_test.MyrialTestCase, FakeData):

    def setUp(self):
        super(TestQueryFunctions, self).setUp()

        self.db.add_function(TestQueryFunctions.test_function)

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
        [x for x in FakeData.emp_table.elements() if x[3] > 25000])

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

    def test_bag_comp_filter_column_compare_ge2(self):
        query = u"""
        emp = SCAN(%s);
        out = [FROM emp WHERE 2 * $1 ≥ $0 EMIT *];
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

    def test_bag_comp_filter_column_compare_le2(self):
        query = u"""
        emp = SCAN(%s);
        out = [FROM emp WHERE $1 ≤ 2 * $0 EMIT *];
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

    def test_bag_comp_filter_column_compare_ne3(self):
        query = u"""
        emp = SCAN(%s);
        out = [FROM emp WHERE $0 // $1 ≠ $1 EMIT *];
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

    def test_explicit_join_unicode(self):
        query = u"""
        emp = SCAN(%s);
        dept = SCAN(%s);
        out = JOIN(emp, dept_id, dept, id);
        out2 = [FROM out EMIT $2 AS emp_name, $5 AS dept_name];
        STORE(out2, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        self.check_result(query, self.join_expected)

    def test_explicit_join(self):
        query = """
        emp = SCAN(%s);
        dept = SCAN(%s);
        out = JOIN(emp, dept_id, dept, id);
        out2 = [FROM out EMIT $2 AS emp_name, $5 AS dept_name];
        STORE(out2, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        self.check_result(query, self.join_expected)

    def test_explicit_join_twocols(self):
        query = """
        query = [1 as dept_id, 25000 as salary];
        emp = SCAN({emp});
        out = JOIN(query, (dept_id, salary), emp, (dept_id, salary));
        out2 = [FROM out EMIT name];
        STORE(out2, OUTPUT);
        """.format(emp=self.emp_key)

        expected = collections.Counter([('Victor Almeida',),
                                        ('Magdalena Balazinska',)])
        self.check_result(query, expected)

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

    def test_two_column_join(self):
        query = """
        D = [1 as dept_id, 25000 as salary];
        out = [FROM D, SCAN({emp}) E
               WHERE E.dept_id == D.dept_id AND E.salary == D.salary
               EMIT E.name AS emp_name];
        STORE(out, OUTPUT);
        """.format(emp=self.emp_key)

        expected = collections.Counter([('Victor Almeida',),
                                        ('Magdalena Balazinska',)])
        self.check_result(query, expected)

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

    def test_join_with_reordering(self):
        # Try both FROM orders of the query and verify they both get the
        # correct answer.
        query = """
        out = [FROM SCAN({d}) AS D, SCAN({e}) E
               WHERE E.dept_id == D.id AND E.salary < 6000
               EMIT E.name, D.id];
        STORE(out, OUTPUT);
        """.format(d=self.dept_key, e=self.emp_key)

        expected = collections.Counter([('Andrew Whitaker', 1),
                                        ('Shumo Chu', 2)])
        self.check_result(query, expected)
        # Swap E and D
        query = """
        out = [FROM SCAN({e}) E, SCAN({d}) AS D
               WHERE E.dept_id == D.id AND E.salary < 6000
               EMIT E.name, D.id];
        STORE(out, OUTPUT);
        """.format(d=self.dept_key, e=self.emp_key)

        expected = collections.Counter([('Andrew Whitaker', 1),
                                        ('Shumo Chu', 2)])
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

    def test_table_literal_boolean(self):
        query = """
        X = [truE as MyTrue, FaLse as MyFalse];
        Y = [FROM scan(%s) as E, X where X.MyTrue emit *];
        STORE(Y, OUTPUT);
        """ % self.emp_key

        res = [x + (True, False) for x in self.emp_table]
        self.check_result(query, collections.Counter(res))

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
        _CONST = [25 AS twenty_five, 1000 AS thousand];

        emp = SCAN(%s);
        out = [FROM emp WHERE salary == *_CONST.twenty_five *
        *_CONST.thousand EMIT *];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000])

        self.check_result(query, expected)

    def test_unbox_from_where_nary_pos(self):
        query = """
        _CONST = [25 AS twenty_five, 1000 AS thousand];

        emp = SCAN(%s);
        out = [FROM emp WHERE salary == *_CONST.$0 *
        *_CONST.$1 EMIT *];
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
        out = [FROM emp, COUNTALL(dept) as size WHERE id > *size EMIT emp.id];
        STORE(out, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        expected = collections.Counter(
            [(x[0],) for x in self.emp_table.elements() if
             x[0] > len(self.dept_table)])
        self.check_result(query, expected)

    def test_inline_table_literal(self):
        query = """
        emp = SCAN(%s);
        dept = SCAN(%s);
        out = [FROM emp, [1,2,3] as tl WHERE id > tl.$2 EMIT emp.id];
        STORE(out, OUTPUT);
        """ % (self.emp_key, self.dept_key)

        expected = collections.Counter(
            [(x[0],) for x in self.emp_table.elements() if
             x[0] > 3])
        self.check_result(query, expected)

    def __aggregate_expected_result(self, apply_func, grouping_col=1,
                                    agg_col=3):
        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[grouping_col]].append(t[agg_col])

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
        self.check_result(query, self.__aggregate_expected_result(avg),
                          test_logical=True)

    def test_stdev(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT STDEV(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        res = self.execute_query(query)
        tp = res.elements().next()
        self.assertAlmostEqual(tp[0], 34001.8006726)

        res = self.execute_query(query, test_logical=True)
        tp = res.elements().next()
        self.assertAlmostEqual(tp[0], 34001.8006726)

    def test_count(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT dept_id, COUNT(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(len))
        self.check_result(query, self.__aggregate_expected_result(len),
                          test_logical=True)

    def test_countall(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT dept_id, COUNTALL()];
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(len))
        self.check_result(query, self.__aggregate_expected_result(len),
                          test_logical=True)

    def test_count_star(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT dept_id, COUNT(*)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(len))
        self.check_result(query, self.__aggregate_expected_result(len),
                          test_logical=True)

    def test_count_star_sql(self):
        query = """
        out = SELECT dept_id, COUNT(*) FROM SCAN(%s) AS X;
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(len))
        self.check_result(query, self.__aggregate_expected_result(len),
                          test_logical=True)

    def test_max_reversed(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT MAX(salary) AS max_salary, dept_id];
        STORE(out, OUTPUT);
        """ % self.emp_key

        ex = self.__aggregate_expected_result(max)
        ex = collections.Counter([(y, x) for (x, y) in ex])
        self.check_result(query, ex)
        self.check_result(query, ex, test_logical=True)

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
        self.check_result(query, expected, test_logical=True)

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
        self.check_result(query, expected, test_logical=True)

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

        with self.assertRaises(raco.myrial.groupby.NonGroupedAccessException):  # noqa
            self.check_result(query, None)

    def test_nested_aggregates_are_illegal(self):
        query = """
        out = [FROM SCAN(%s) AS X
               EMIT id+dept_id, MIN(53 + MAX(salary)) AS foo];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(NestedAggregateException):
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

        with self.assertRaises(ColumnIndexOutOfBounds):
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

    def test_md5(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT id, md5(name)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        def md5_as_long(x):
            m = md5.new()
            m.update(x)
            return int(m.hexdigest(), 16) >> 64

        expected = collections.Counter(
            [(x[0], md5_as_long(x[2])) for x in self.emp_table.elements()])
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

        with self.assertRaises(NoSuchRelationException):
            self.check_result(query, collections.Counter())

    def test_bad_relation_name(self):
        query = """
        y = empty(a:int);
        z = [from s y      -- bug: s does not exist
             emit y.a];
        store(z, debug);
        """

        with self.assertRaises(NoSuchRelationException):
            self.check_result(query, collections.Counter())

    def test_bad_alias(self):
        query = """
        y = empty(a:int);
        z = [from y s      -- bug: extra s
             emit y.a];
        store(z, debug);
        """

        with self.assertRaises(NoSuchRelationException):
            self.check_result(query, collections.Counter())

    def test_bad_alias_wildcard(self):
        query = """
        y = empty(a:int);
        z = [from y s      -- bug: errant s
             emit y.*];
        store(z, debug);
        """

        with self.assertRaises(NoSuchRelationException):
            self.check_result(query, collections.Counter())

    def test_scan_error(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT id, !!!FROG(val)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(MyrialCompileException):
            self.check_result(query, collections.Counter())

    def test_relation_scope_error(self):
        query = """
        out = [FROM EMPTY(x:INT) AS X EMIT z.*];
        STORE(out, OUTPUT);
        """

        with self.assertRaises(NoSuchRelationException):
            self.check_result(query, collections.Counter())

    def test_relation_scope_error2(self):
        query = """
        z = EMPTY(z:INT);
        out = [FROM EMPTY(x:INT) AS X EMIT z.*];
        STORE(out, OUTPUT);
        """

        with self.assertRaises(NoSuchRelationException):
            self.check_result(query, collections.Counter())

    def test_parse_error(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT $(val)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(MyrialCompileException):
            self.check_result(query, collections.Counter())

    def test_no_such_udf(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT FooFunction(X.salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(NoSuchFunctionException):
            self.check_result(query, collections.Counter())

    def test_reserved_udf(self):
        query = """
        DEF avg(x, y): (x + y) / 2;
        out = [FROM SCAN(%s) AS X EMIT avg(X.salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(ReservedTokenException):
            self.check_result(query, collections.Counter())

    def test_duplicate_udf(self):
        query = """
        DEF foo(x, y): x + y;
        DEF bar(): 7;
        DEF foo(x): -1 * x;

        out = [FROM SCAN(%s) AS X EMIT foo(X.salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(DuplicateFunctionDefinitionException):
            self.check_result(query, collections.Counter())

    def test_invalid_argument_udf(self):
        query = """
        DEF Foo(x, y): cos(x) * sin(y);
        out = [FROM SCAN(%s) AS X EMIT Foo(X.salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(InvalidArgumentList):
            self.check_result(query, collections.Counter())

    def test_undefined_variable_udf(self):
        query = """
        DEF Foo(x, y): cos(x) * sin(z);
        out = [FROM SCAN(%s) AS X EMIT Foo(X.salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(UndefinedVariableException):
            self.check_result(query, collections.Counter())

    def test_duplicate_variable_udf(self):
        query = """
        DEF Foo(x, x): cos(x) * sin(x);
        out = [FROM SCAN(%s) AS X EMIT Foo(X.salary, X.dept_id)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(DuplicateVariableException):
            self.check_result(query, collections.Counter())

    def test_nary_udf(self):
        query = """
        DEF Foo(a,b): [a + b, a - b];

        out = [FROM SCAN(%s) AS X EMIT id, Foo(salary, dept_id) as [x, y]];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter([(t[0], t[1] + t[3], t[3] - t[1])
                                        for t in self.emp_table])
        self.check_result(query, expected)

    def test_nary_udf_name_count(self):
        query = """
        DEF Foo(a,b): [a + b, a - b];

        out = [FROM SCAN(%s) AS X EMIT id, Foo(salary, dept_id) as [x, y, z]];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(IllegalColumnNamesException):
            self.check_result(query, None)

    def test_nary_udf_illegal_nesting(self):
        query = """
        DEF Foo(x): [x + 3, x - 3];
        DEF Bar(a,b): [Foo(x), Foo(b)];

        out = [FROM SCAN(%s) AS X EMIT id, Bar(salary, dept_id) as [x, y]];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(NestedTupleExpressionException):
            self.check_result(query, None)

    def test_nary_udf_illegal_wildcard(self):
        query = """
        DEF Foo(x): [x + 3, *];

        out = [FROM SCAN(%s) AS X EMIT id, Foo(salary, dept_id) as [x, y]];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(IllegalWildcardException):
            self.check_result(query, None)

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

    def test_const(self):
        expr = "30 + 15 // 7 + -45"

        query = """
        CONST myconstant: %s;

        out = [myconstant AS t];
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

    def test_least_function(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT least(X.id,X.dept_id,1)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(min(t[0], t[1], 1),)
             for t in self.emp_table])
        self.check_result(query, expected)

    def test_greatest_function(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT greatest(X.id,X.dept_id,3)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(max(t[0], t[1], 3),)
             for t in self.emp_table])
        self.check_result(query, expected)

    def test_lesser_function(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT lesser(X.id,X.dept_id)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(min(t[0], t[1]),)
             for t in self.emp_table])
        self.check_result(query, expected)

    def test_greater_function(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT greater(X.id,X.dept_id)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(max(t[0], t[1],),)
             for t in self.emp_table])
        self.check_result(query, expected)

    def test_uda_illegal_init(self):
        query = """
        uda Foo(x,y) {
            [0 as A, *];
            [A + x, A + y];
             A;
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, Foo(salary, id)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(IllegalWildcardException):
            self.check_result(query, None)

    def test_uda_illegal_update(self):
        query = """
        uda Foo(x,y) {
            [0 as A, 1 as B];
            [A + x + y, *];
             A + B;
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, Foo(salary, id)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(MyrialCompileException):
            self.check_result(query, None)

    def test_uda_nested_emitter(self):
        query = """
        uda Foo(x) {
            [0 as A];
            [A + x];
            [A];
        };
        uda Bar(x) {
            [0 as B];
            [B + x];
            Foo(B);
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, Bar(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(NestedAggregateException):
            self.check_result(query, None)

    def test_uda_nested_init(self):
        query = """
        uda Foo(x) {
            [0 as A];
            [A + x];
            [A];
        };
        uda Bar(x) {
            [Foo(0) as B];
            [B + x];
            B;
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, Bar(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(NestedAggregateException):
            self.check_result(query, None)

    def test_uda_nested_update(self):
        query = """
        uda Foo(x) {
            [0 as A];
            [A + x];
            [A];
        };
        uda Bar(x) {
            [0 as B];
            [Foo(B)];
            B;
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, Bar(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(NestedAggregateException):
            self.check_result(query, None)

    def test_uda_unary_emit_arg_list(self):
        query = """
        uda MyAvg(val) {
            [0 as _sum, 0 as _count];
            [_sum + val, _count + 1];
            [_sum / _count];
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, MyAvg(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        def agg_func(x):
            return float(sum(x)) / len(x)

        expected = self.__aggregate_expected_result(agg_func)
        self.check_result(query, expected)

    def test_second_max_uda(self):
        """UDA to compute the second largest element in a collection."""
        query = """
        uda SecondMax(val) {
            [0 as _max, 0 as second_max];
            [case when val > _max then val else _max end,
             case when val > _max then _max when val > second_max then val
             else second_max end];
             second_max;
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, SecondMax(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        def agg_func(x):
            if len(x) < 2:
                return 0
            else:
                return sorted(x, reverse=True)[1]

        expected = self.__aggregate_expected_result(agg_func)
        self.check_result(query, expected)

    def test_multi_invocation_uda(self):
        query = """
        uda MaxDivMin(val) {
            [9999999 as _min, 0 as _max];
            [case when val < _min then val else _min end,
             case when val > _max then val else _max end];
             _max / _min;
        };

        out = [FROM SCAN(%s) AS X EMIT
               MaxDivMin(id) + dept_id + MaxDivMin(salary), dept_id];
        STORE(out, OUTPUT);
        """ % self.emp_key

        d = collections.defaultdict(list)
        for t in self.emp_table.elements():
            d[t[1]].append(t)

        results = []
        for k, tpls in d.iteritems():
            max_salary = max(t[3] for t in tpls)
            min_salary = min(t[3] for t in tpls)
            max_id = max(t[0] for t in tpls)
            min_id = min(t[0] for t in tpls)
            results.append((k + float(max_salary) / min_salary +
                            float(max_id) / min_id, k))

        self.check_result(query, collections.Counter(results))

    def test_multiple_uda(self):
        query = """
        uda MyMax1(val) {
            [0 as _max];
            [case when val > _max then val else _max end];
             _max;
        };
        uda MyMax2(val) {
            [0 as _max];
            [case when val > _max then val else _max end];
             _max;
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, MyMax1(salary), MyMax2(id)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        d = collections.defaultdict(list)
        for t in self.emp_table.elements():
            d[t[1]].append(t)

        results = []
        for k, tpls in d.iteritems():
            max_salary = max(t[3] for t in tpls)
            max_id = max(t[0] for t in tpls)
            results.append((k, max_salary, max_id))

        self.check_result(query, collections.Counter(results))

    def test_uda_no_emit_clause(self):
        query = """
        uda MyCount() {
            [0 as _count];
            [_count + 1];
        };
        out = [FROM SCAN(%s) AS X EMIT dept_id, MyCount()];
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(len))

    def test_uda_no_emit_clause_many_cols(self):
        query = """
        uda MyAggs(x) {
            [0 as _count, 0 as _sum, 0 as _sumsq];
            [_count + 1, _sum + x, _sumsq + x*x];
        };
        out = [FROM SCAN(%s) AS X EMIT MyAggs(salary) as [a, b, c]];
        STORE(out, OUTPUT);
        """ % self.emp_key

        c = len(list(self.emp_table.elements()))
        s = sum(d for a, b, c, d in self.emp_table.elements())
        sq = sum(d * d for a, b, c, d in self.emp_table.elements())
        expected = collections.Counter([(c, s, sq)])
        self.check_result(query, expected)

        # Test with two different column orders in case the undefined
        # order used by Python is correct by chance.
        query = """
        uda MyAggs(x) {
            [0 as _count, 0 as _sumsq, 0 as _sum];
            [_count + 1, _sumsq + x*x, _sum + x];
        };
        out = [FROM SCAN(%s) AS X EMIT MyAggs(salary) as [a, b, c]];
        STORE(out, OUTPUT);
        """ % self.emp_key

        c = len(list(self.emp_table.elements()))
        sq = sum(d * d for a, b, c, d in self.emp_table.elements())
        s = sum(d for a, b, c, d in self.emp_table.elements())
        expected = collections.Counter([(c, sq, s)])
        self.check_result(query, expected)

    def test_uda_with_udf(self):
        query = """
        def foo(x, y): x + y;
        uda max2(x, y) {
            [0 as _max];
            [case when foo(x, y) > _max then foo(x, y) else _max end];
            _max;
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, max2(salary, id)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        d = collections.defaultdict(list)
        for t in self.emp_table.elements():
            d[t[1]].append(t)

        results = []
        for k, tpls in d.iteritems():
            results.append((k, max(t[3] + t[0] for t in tpls)))

        self.check_result(query, collections.Counter(results))

    def test_uda_with_subsequent_project_0(self):
        query = """
        def foo(x, y): x + y;
        uda max2(x, y) {
            [0 as _max];
            [case when foo(x, y) > _max then foo(x, y) else _max end];
            _max;
        };

        inter = [FROM SCAN(%s) AS X EMIT dept_id, max2(salary, id)];
        out = [from inter emit $0];
        STORE(out, OUTPUT);
        """ % self.emp_key

        d = collections.defaultdict(list)
        for t in self.emp_table.elements():
            d[t[1]].append(t)

        results = []
        for k, tpls in d.iteritems():
            results.append((k, max(t[3] + t[0] for t in tpls)))
        results = [(t[0],) for t in results]

        self.check_result(query, collections.Counter(results))

    def test_uda_with_subsequent_project_1(self):
        query = """
        def foo(x, y): x + y;
        uda max2(x, y) {
            [0 as _max];
            [case when foo(x, y) > _max then foo(x, y) else _max end];
            _max;
        };

        inter = [FROM SCAN(%s) AS X EMIT dept_id, max2(salary, id)];
        out = [from inter emit $1];
        STORE(out, OUTPUT);
        """ % self.emp_key

        d = collections.defaultdict(list)
        for t in self.emp_table.elements():
            d[t[1]].append(t)

        results = []
        for k, tpls in d.iteritems():
            results.append((k, max(t[3] + t[0] for t in tpls)))
        results = [(t[1],) for t in results]

        self.check_result(query, collections.Counter(results))

    def test_uda_with_subsequent_project_2(self):
        query = """
        def foo(x, y): x + y;
        uda max2(x, y) {
            [0 as _max];
            [case when foo(x, y) > _max then foo(x, y) else _max end];
            _max;
        };

        inter = [FROM SCAN(%s) AS X EMIT dept_id, max2(salary, id)
                                       , max2(dept_id, id)];
        out = [from inter emit $1];
        STORE(out, OUTPUT);
        """ % self.emp_key

        d = collections.defaultdict(list)
        for t in self.emp_table.elements():
            d[t[1]].append(t)

        results = []
        for k, tpls in d.iteritems():
            results.append((k,
                            max(t[3] + t[0] for t in tpls),
                            max(t[1] + t[0] for t in tpls)))
        results = [(t[1],) for t in results]

        self.check_result(query, collections.Counter(results))

    def __run_multiple_emitter_test(self, include_column_names):

        if include_column_names:
            names = " AS [mysum, mycount, myavg]"
        else:
            names = ""

        query = """
        uda SumCountMean(x) {
          [0 as _sum, 0 as _count];
          [_sum + x, _count + 1];
          [_sum, _count, _sum/_count];
        };
        out = [FROM SCAN(%s) AS X EMIT dept_id, SumCountMean(salary) %s,
               dept_id+3, max(id) as max_id];
        STORE(out, OUTPUT);
        """ % (self.emp_key, names)

        d = collections.defaultdict(list)
        for t in self.emp_table.elements():
            d[t[1]].append(t)

        results = []
        for k, tpls in d.iteritems():
            _sum = sum(x[3] for x in tpls)
            _count = len(tpls)
            _avg = float(_sum) / _count
            _max_id = max(x[0] for x in tpls)
            results.append((k, _sum, _count, _avg, k + 3, _max_id))

        self.check_result(query, collections.Counter(results))

    def test_uda_multiple_emitters_default_names(self):
        self.__run_multiple_emitter_test(False)

    def test_uda_multiple_emitters_provided_names(self):
        self.__run_multiple_emitter_test(True)

        scheme_actual = self.db.get_scheme('OUTPUT')
        scheme_expected = scheme.Scheme([
            ('dept_id', types.LONG_TYPE), ('mysum', types.LONG_TYPE),
            ('mycount', types.LONG_TYPE), ('myavg', types.FLOAT_TYPE),
            ('_COLUMN4_', types.LONG_TYPE), ('max_id', types.LONG_TYPE)])

        self.assertEquals(scheme_actual, scheme_expected)

    def test_emit_arg_bad_column_name_length(self):
        query = """

        out = [FROM SCAN(%s) AS X EMIT dept_id AS [dept_id1, dept_id2]];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(IllegalColumnNamesException):
            self.check_result(query, None)

    def test_uda_bad_column_name_length(self):
        query = """
        uda Fubar(x, y, z) {
          [0 as Q];
          [Q + 1];
          [1,2,3];
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, Fubar(1, salary, id)
               AS [A, B, C, D]];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(IllegalColumnNamesException):
            self.check_result(query, None)

    def test_uda_init_tuple_valued(self):
        query = """
        uda Foo(x) {
          [0 as Q];
          [Q + 1];
          [1,2,3];
        };

        uda Bar(x) {
          [Foo(0) as [A, B, C]];
          [Q * 8];
          [1,2,3];
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, Bar(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(NestedTupleExpressionException):
            self.check_result(query, None)

    def test_uda_update_tuple_valued(self):
        query = """
        uda Foo(x) {
          [0 as Q];
          [Q + 1];
          [1,2,3];
        };

        uda Bar(x) {
          [0 as Q];
          [Foo(Q + 1)];
          [1,2,3];
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, Bar(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(NestedTupleExpressionException):
            self.check_result(query, None)

    def test_uda_result_tuple_valued(self):
        query = """
        uda Foo(x) {
          [0 as Q];
          [Q + 1];
          [1,2,3];
        };

        uda Bar(x) {
          [0 as Q];
          [Q + 2];
          [1,2,Foo(3)];
        };

        out = [FROM SCAN(%s) AS X EMIT dept_id, Bar(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(NestedTupleExpressionException):
            self.check_result(query, None)

    def test_uda_multiple_emitters_nested(self):
        """Test that we raise an Exception if a tuple-valued UDA doesn't appear
        by itself in an emit expression."""
        query = """
        uda SumCountMean(x) {
          [0 as _sum, 0 as _count];
          [_sum + x, _count + 1];
          [_sum, _count, _sum/_count];
        };
        out = [FROM SCAN(%s) AS X EMIT dept_id, SumCountMean(salary) + 5];
        STORE(out, OUTPUT);
        """ % self.emp_key

        with self.assertRaises(NestedTupleExpressionException):
            self.check_result(query, None)

    __DECOMPOSED_UDA = """
        uda LogicalAvg(x) {
          [0 as _sum, 0 as _count];
          [_sum + x, _count + 1];
          float(_sum); -- Note bogus return value
        };
        uda LocalAvg(x) {
          [0 as _sum, 0 as _count];
          [_sum + x, _count + 1];
        };
        uda RemoteAvg(_local_sum, _local_count) {
          [0 as _sum, 0 as _count];
          [_sum + _local_sum, _count + _local_count];
          [_sum/_count];
        };
        uda* LogicalAvg {LocalAvg, RemoteAvg};
    """

    __ARG_MAX_UDA = """
        def pickval(id, salary, val, _id, _salary, _val):
           case when salary > _salary then val
                when salary = _salary and id > _id then val
                else _val end;
        uda ArgMax(id, dept_id, name, salary) {
          [0 as _id, 0 as _dept_id, "" as _name, 0 as _salary];
          [pickval(id, salary, id, _id, _salary, _id),
           pickval(id, salary, dept_id, _id, _salary, _dept_id),
           pickval(id, salary, name, _id, _salary, _name),
           pickval(id, salary, salary, _id, _salary, _salary)];
          [_id, _dept_id, _name, _salary];
        };
    """

    __ARG_MAX_UDA_UNNECESSARY_EXPR = """
        def pickval(id, salary, val, _id, _salary, _val):
           case when salary > _salary then val
                when salary = _salary and id > _id then val
                else _val end;
        uda ArgMax(id, dept_id, name, salary) {
          [0 as _id, 0 as _dept_id, "" as _name, 0 as _salary];
          [pickval(id, salary, greater(id, id), _id, _salary, _id),
           pickval(id, salary, lesser(dept_id, dept_id), _id, _salary,
                   _dept_id),
           pickval(id, salary, case when name="" then name else name end, _id,
                   _salary, _name),
           pickval(id, salary, salary * 1, _id, _salary, _salary)];
          [_id, _dept_id, _name, _salary];
        };
    """

    def test_decomposable_average_uda(self):
        """Test of a decomposed average UDA.

        Note that the logical aggregate returns a broken value, so
        this test only passes if we decompose the aggregate properly.
        """

        query = """%s
        out = [FROM SCAN(%s) AS X EMIT dept_id, LogicalAvg(salary)];
        STORE(out, OUTPUT);
        """ % (TestQueryFunctions.__DECOMPOSED_UDA, self.emp_key)

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[1]].append(t[3])

        tuples = []
        for key, vals in result_dict.iteritems():
            _cnt = len(vals)
            _sum = sum(vals)
            tuples.append((key, float(_sum) / _cnt))

        self.check_result(query, collections.Counter(tuples))

    def test_decomposable_nary_uda(self):

        query = """
        uda Sum2(x, y) {
          [0 as sum_x, 0 as sum_y];
          [sum_x + x, sum_y + y];
        };
        uda* Sum2 {Sum2, Sum2};
        out = [FROM SCAN(%s) AS X EMIT
               Sum2(id, salary) AS [id_sum, salary_sum]];
        STORE(out, OUTPUT);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)

        for t in self.emp_table.elements():
            result_dict[t[1]].append(t)

        id_sum = sum(t[0] for t in self.emp_table.elements())
        salary_sum = sum(t[3] for t in self.emp_table.elements())

        tuples = [(id_sum, salary_sum)]
        self.check_result(query, collections.Counter(tuples))

    def test_arg_max_uda(self):
        """Test of an arg_max UDA.
        """

        query = """
        {arg}
        emp = scan({emp});
        out = [from emp emit ArgMax(id, dept_id, name, salary)];
        store(out, OUTPUT);
        """.format(arg=self.__ARG_MAX_UDA, emp=self.emp_key)

        tuples = [(a, b, c, d) for (a, b, c, d) in self.emp_table
                  if all(d > d1 or d == d1 and a >= a1
                         for a1, b1, c1, d1 in self.emp_table)]
        self.check_result(query, collections.Counter(tuples))

    def test_arg_max_uda_with_references(self):
        """Test of an arg_max UDA with named, unnamed, and dotted
        attribute references.
        """

        query = """
        {arg}
        emp = scan({emp});
        out = [from emp emit ArgMax(id, emp.dept_id, $2, emp.$3)];
        store(out, OUTPUT);
        """.format(arg=self.__ARG_MAX_UDA, emp=self.emp_key)

        tuples = [(a, b, c, d) for (a, b, c, d) in self.emp_table
                  if all(d > d1 or d == d1 and a >= a1
                         for a1, b1, c1, d1 in self.emp_table)]
        self.check_result(query, collections.Counter(tuples))

    def test_arg_max_uda_with_functions(self):
        """Test of an arg_max UDA with expressions as inputs.
        """

        query = """
        {arg}
        emp = scan({emp});
        out = [from emp emit ArgMax(id,
                        greater(dept_id, dept_id),
                        case when id=1 then name else name end,
                        salary)];
        store(out, OUTPUT);
        """.format(arg=self.__ARG_MAX_UDA, emp=self.emp_key)

        tuples = [(a, b, c, d) for (a, b, c, d) in self.emp_table
                  if all(d > d1 or d == d1 and a >= a1
                         for a1, b1, c1, d1 in self.emp_table)]
        self.check_result(query, collections.Counter(tuples))

    def test_decomposable_arg_max_uda(self):
        """Test of a decomposable arg_max UDA.
        """

        query = """
        {arg}
        uda* ArgMax {{ArgMax, ArgMax}};
        emp = scan({emp});
        out = [from emp emit ArgMax(id, dept_id, name, salary)
               as [a, b, c, d]];
        store(out, OUTPUT);
        """.format(arg=self.__ARG_MAX_UDA, emp=self.emp_key)

        tuples = [(a, b, c, d) for (a, b, c, d) in self.emp_table
                  if all(d > d1 or d == d1 and a >= a1
                         for a1, b1, c1, d1 in self.emp_table)]
        self.check_result(query, collections.Counter(tuples))

        """Test of an arg_max UDA with named, unnamed, and dotted
        attribute references.
        """

    def test_decomposable_arg_max_uda_with_references(self):
        """Test of a decomposable arg_max UDA with named, unnamed, and dotted
        attribute references.
        """
        query = """
        {arg}
        uda* ArgMax {{ArgMax, ArgMax}};
        emp = scan({emp});
        out = [from emp emit ArgMax(id, emp.dept_id, $2, emp.$3)
               as [a, b, c, d]];
        store(out, OUTPUT);
        """.format(arg=self.__ARG_MAX_UDA, emp=self.emp_key)

        tuples = [(a, b, c, d) for (a, b, c, d) in self.emp_table
                  if all(d > d1 or d == d1 and a >= a1
                         for a1, b1, c1, d1 in self.emp_table)]
        self.check_result(query, collections.Counter(tuples))

    def test_decomposable_arg_max_uda_with_functions(self):
        """Test of a decomposable arg_max UDA with expressions as inputs.
        """

        query = """
        {arg}
        uda* ArgMax {{ArgMax, ArgMax}};
        emp = scan({emp});
        out = [from emp emit ArgMax(id,
                        greater(dept_id, dept_id),
                        case when id=1 then name else name end,
                        salary)];
        store(out, OUTPUT);
        """.format(arg=self.__ARG_MAX_UDA, emp=self.emp_key)

        tuples = [(a, b, c, d) for (a, b, c, d) in self.emp_table
                  if all(d > d1 or d == d1 and a >= a1
                         for a1, b1, c1, d1 in self.emp_table)]

        self.check_result(query, collections.Counter(tuples))

    def test_arg_max_uda_internal_exprs(self):
        """Test of an arg_max UDA.
        """

        query = """
        {arg}
        emp = scan({emp});
        out = [from emp emit ArgMax(id, dept_id, name, salary)];
        store(out, OUTPUT);
        """.format(arg=self.__ARG_MAX_UDA_UNNECESSARY_EXPR, emp=self.emp_key)

        tuples = [(a, b, c, d) for (a, b, c, d) in self.emp_table
                  if all(d > d1 or d == d1 and a >= a1
                         for a1, b1, c1, d1 in self.emp_table)]
        self.check_result(query, collections.Counter(tuples))

    def test_arg_max_uda_internal_exprs_with_references(self):
        """Test of an arg_max UDA with named, unnamed, and dotted
        attribute references.
        """

        query = """
        {arg}
        emp = scan({emp});
        out = [from emp emit ArgMax(id, emp.dept_id, $2, emp.$3)];
        store(out, OUTPUT);
        """.format(arg=self.__ARG_MAX_UDA_UNNECESSARY_EXPR, emp=self.emp_key)

        tuples = [(a, b, c, d) for (a, b, c, d) in self.emp_table
                  if all(d > d1 or d == d1 and a >= a1
                         for a1, b1, c1, d1 in self.emp_table)]
        self.check_result(query, collections.Counter(tuples))

    def test_arg_max_uda_internal_exprs_with_functions(self):
        """Test of an arg_max UDA with expressions as inputs.
        """

        query = """
        {arg}
        emp = scan({emp});
        out = [from emp emit ArgMax(id,
                        greater(dept_id, dept_id),
                        case when id=1 then name else name end,
                        salary)];
        store(out, OUTPUT);
        """.format(arg=self.__ARG_MAX_UDA_UNNECESSARY_EXPR, emp=self.emp_key)

        tuples = [(a, b, c, d) for (a, b, c, d) in self.emp_table
                  if all(d > d1 or d == d1 and a >= a1
                         for a1, b1, c1, d1 in self.emp_table)]
        self.check_result(query, collections.Counter(tuples))

    def test_decomposable_arg_max_uda_internal_exprs(self):
        """Test of a decomposable arg_max UDA.
        """

        query = """
        {arg}
        uda* ArgMax {{ArgMax, ArgMax}};
        emp = scan({emp});
        out = [from emp emit ArgMax(id, dept_id, name, salary)
               as [a, b, c, d]];
        store(out, OUTPUT);
        """.format(arg=self.__ARG_MAX_UDA_UNNECESSARY_EXPR, emp=self.emp_key)

        tuples = [(a, b, c, d) for (a, b, c, d) in self.emp_table
                  if all(d > d1 or d == d1 and a >= a1
                         for a1, b1, c1, d1 in self.emp_table)]
        self.check_result(query, collections.Counter(tuples))

        """Test of an arg_max UDA with named, unnamed, and dotted
        attribute references.
        """

    def test_decomposable_arg_max_uda_internal_exprs_with_references(self):
        """Test of a decomposable arg_max UDA with named, unnamed, and dotted
        attribute references.
        """
        query = """
        {arg}
        uda* ArgMax {{ArgMax, ArgMax}};
        emp = scan({emp});
        out = [from emp emit ArgMax(id, emp.dept_id, $2, emp.$3)
               as [a, b, c, d]];
        store(out, OUTPUT);
        """.format(arg=self.__ARG_MAX_UDA_UNNECESSARY_EXPR, emp=self.emp_key)

        tuples = [(a, b, c, d) for (a, b, c, d) in self.emp_table
                  if all(d > d1 or d == d1 and a >= a1
                         for a1, b1, c1, d1 in self.emp_table)]
        self.check_result(query, collections.Counter(tuples))

    def test_decomposable_arg_max_uda_internal_exprs_with_functions(self):
        """Test of a decomposable arg_max UDA with expressions as inputs.
        """

        query = """
        {arg}
        uda* ArgMax {{ArgMax, ArgMax}};
        emp = scan({emp});
        out = [from emp emit ArgMax(id,
                        greater(dept_id, dept_id),
                        case when id=1 then name else name end,
                        salary)];
        store(out, OUTPUT);
        """.format(arg=self.__ARG_MAX_UDA_UNNECESSARY_EXPR, emp=self.emp_key)

        tuples = [(a, b, c, d) for (a, b, c, d) in self.emp_table
                  if all(d > d1 or d == d1 and a >= a1
                         for a1, b1, c1, d1 in self.emp_table)]
        self.check_result(query, collections.Counter(tuples))

    def test_decomposable_average_uda_repeated(self):
        """Test of repeated invocations of decomposed UDAs."""

        query = """%s
        out = [FROM SCAN(%s) AS X EMIT dept_id,
               LogicalAvg(salary) + LogicalAvg($0)];
        STORE(out, OUTPUT);
        """ % (TestQueryFunctions.__DECOMPOSED_UDA, self.emp_key)

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[1]].append(t)

        tuples = []
        for key, vals in result_dict.iteritems():
            _cnt = len(vals)
            _salary_sum = sum(t[3] for t in vals)
            _id_sum = sum(t[0] for t in vals)
            tuples.append((key, (float(_salary_sum) + float(_id_sum)) / _cnt))

        self.check_result(query, collections.Counter(tuples))

    def test_decomposable_sum_uda(self):
        """Test of a decomposed sum UDA.

        Note that the logical aggregate returns a broken value, so
        this test only passes if we decompose the aggregate properly.
        """

        query = """
        uda MySumBroken(x) {
          [0 as _sum];
          [_sum + x];
          17; -- broken
        };
        uda MySum(x) {
          [0 as _sum];
          [_sum + x];
        };
        uda* MySumBroken {MySum, MySum};

        out = [FROM SCAN(%s) AS X EMIT dept_id, MySumBroken(salary)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        self.check_result(query, self.__aggregate_expected_result(sum))

    def test_decomposable_uda_with_builtin_agg(self):
        """Test of a decomposed UDA + builtin aggregate.

        Note that the logical aggregate returns a broken value, so
        this test only passes if we decompose the aggregate properly.
        """

        query = """
        uda MySumBroken(x) {
          [0 as _sum];
          [_sum + x];
          17; -- broken
        };
        uda MySum(x) {
          [0 as _sum];
          [_sum + x];
        };
        uda* MySumBroken {MySum, MySum};

        out = [FROM SCAN(%s) AS X EMIT dept_id, MySumBroken(salary), SUM(id)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[1]].append(t)

        tuples = []
        for key, vals in result_dict.iteritems():
            _salary_sum = sum(t[3] for t in vals)
            _id_sum = sum(t[0] for t in vals)
            tuples.append((key, _salary_sum, _id_sum))

        self.check_result(query, collections.Counter(tuples))

    def test_duplicate_decomposable_uda(self):
        query = """
        uda Agg1(x) {
          [0 as _sum];
          [_sum + x];
        };

        uda* Agg1 {Agg1, Agg1};
        uda* Agg1 {Agg1, Agg1};
        """

        with self.assertRaises(DuplicateFunctionDefinitionException):
            self.check_result(query, None)

    def test_decomposable_uda_type_check_fail1(self):
        query = """
        uda Logical(x) {
          [0 as _sum];
          [_sum + x];
        };
        uda Local(x, y) {
          [0 as _sum];
          [_sum + x];
        };
        uda* Logical {Local, Logical};
        """

        with self.assertRaises(InvalidArgumentList):
            self.check_result(query, None)

    def test_decomposable_uda_type_check_fail2(self):
        query = """
        uda Logical(x) {
          [0 as _sum];
          [_sum + x];
        };
        uda Remote(x, y) {
          [0 as _sum];
          [_sum + x];
        };
        uda* Logical {Logical, Remote};
        """

        with self.assertRaises(InvalidArgumentList):
            self.check_result(query, None)

    def test_decomposable_uda_type_check_fail3(self):
        query = """
        uda Logical(x) {
          [0 as _sum];
          [_sum + x];
        };
        uda Remote(x) {
          [0 as _sum];
          [_sum + x];
          [1, 2, 3];
        };
        uda* Logical {Logical, Remote};
        """

        with self.assertRaises(InvalidEmitList):
            self.check_result(query, None)

    def test_running_mean_sapply(self):
        query = """
        APPLY RunningMean(value) {
            [0 AS _count, 0 AS _sum];
            [_count + 1, _sum + value];
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
            [_sum + x];
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
        with self.assertRaises(MyrialCompileException):
            self.check_result(";", None)

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

    def test_flip_zero(self):
        """flip(0) should always evaluate to false"""
        query = """
        X = [FROM SCAN(%s) AS X WHERE flip(0) EMIT *];
        STORE(X, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter()
        self.check_result(query, expected)

    def test_flip_one(self):
        """flip(1) should always evaluate to true"""
        query = """
        X = [FROM SCAN(%s) AS X WHERE flip(1) EMIT *];
        STORE(X, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(self.emp_table.elements())
        self.check_result(query, expected)

    def test_substr(self):
        query = """
        ZERO = [0];
        THREE = [3];
        out = [FROM SCAN(%s) AS X EMIT X.id, substr(X.name, *ZERO, *THREE)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[0], x[2][0:3]) for x in self.emp_table.elements()])
        self.check_result(query, expected)

    def test_len(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT X.id, len(X.name)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[0], len(x[2])) for x in self.emp_table.elements()])
        self.check_result(query, expected)

    def test_head(self):
        query = """
        out = [FROM SCAN(%s) AS X EMIT X.id, head(X.name, 10)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[0], x[2][0:10]) for x in self.emp_table.elements()])
        self.check_result(query, expected)

    def test_tail(self):
        query = """
        ZERO = [0];
        THREE = [3];
        out = [FROM SCAN(%s) AS X EMIT X.id, tail(X.name, 10)];
        STORE(out, OUTPUT);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[0], (lambda i: i if len(i) <= 10 else i[len(i) - 10:])(x[2]))
                for x in self.emp_table.elements()])
        self.check_result(query, expected)

    def test_column_name_reserved(self):
        query = """
        T = EMPTY(x:INT);
        A = [FROM T EMIT SafeDiv(x, 3) AS SafeDiv];
        STORE (A, BadProgram);
        """
        with self.assertRaises(ReservedTokenException):
            self.check_result(query, None)

    def test_bug_226(self):
        query = """
        T = scan({emp});
        A = select id, salary from T where 1=1;
        B = select id, salary from A where salary=90000;
        C = select A.* from B, A where A.salary < B.salary;
        STORE (C, OUTPUT);
        """.format(emp=self.emp_key)

        expected = collections.Counter(
            (i, s) for (i, d, n, s) in self.emp_table
            for (i2, d2, n2, s2) in self.emp_table
            if s2 == 90000 and s < s2)

        self.assertEquals(expected, self.execute_query(query))

    def test_column_mixed_case_reserved(self):
        query = """
        T = EMPTY(x:INT);
        A = [FROM T EMIT MAX(x) AS maX];
        STORE (A, BadProgram);
        """
        with self.assertRaises(ReservedTokenException):
            self.check_result(query, None)

    def test_variable_name_reserved(self):
        query = """
        T = EMPTY(x:INT);
        avg = COUNTALL(T);
        STORE (countall, BadProgram);
        """
        with self.assertRaises(ReservedTokenException):
            self.check_result(query, None)

    def test_empty_query(self):
        query = """
        T1 = empty(x:INT);
        """
        with self.assertRaises(MyrialCompileException):
            self.check_result(query, None)

    def test_sink(self):
        query = """
        ZERO = [0];
        A = [from ZERO emit *];
        SINK(A);
        """
        self.evaluate_sink_query(query)

    def test_string_cast(self):
        query = """
        emp = SCAN(%s);
        bc = [FROM emp EMIT STRING(emp.dept_id) AS foo];
        STORE(bc, OUTPUT);
        """ % self.emp_key

        ex = collections.Counter((str(d),) for (i, d, n, s) in self.emp_table)
        ex_scheme = scheme.Scheme([('foo', types.STRING_TYPE)])
        self.check_result(query, ex)

    def test_float_cast(self):
        query = """
        emp = SCAN(%s);
        bc = [FROM emp EMIT float(emp.dept_id) AS foo];
        STORE(bc, OUTPUT);
        """ % self.emp_key

        ex = collections.Counter((float(d),) for (i, d, n, s) in self.emp_table)  # noqa
        ex_scheme = scheme.Scheme([('foo', types.DOUBLE_TYPE)])
        self.check_result(query, ex, ex_scheme)

    def test_sequence(self):
        query = """
        T1 = scan({rel});
        store(T1, OUTPUT);
        T2 = scan({rel});
        store(T2, OUTPUT2);
        """.format(rel=self.emp_key)

        physical_plan = self.get_physical_plan(query)
        self.assertIsInstance(physical_plan, raco.algebra.Sequence)
        self.check_result(query, self.emp_table, output='OUTPUT')
        self.check_result(query, self.emp_table, output='OUTPUT2')

    def test_238_dont_renumber_columns(self):
        # see https://github.com/uwescience/raco/issues/238
        query = """
        x = [1 as a, 2 as b];
        y = [from x as x1, x as x2
             emit x2.a, x2.b];
        z = [from y emit a];
        store(z, OUTPUT);"""

        self.check_result(query, collections.Counter([(1,)]))

    def test_implicit_column_names(self):
        query = """
        x = [1 as a, 2 as b];
        y = [from x as x1, x as x2
             emit $0, $1];
        store(y, OUTPUT);"""

        expected_scheme = scheme.Scheme([('a', types.LONG_TYPE),
                                         ('b', types.LONG_TYPE)])
        self.check_result(query, collections.Counter([(1, 2)]),
                          scheme=expected_scheme)

    def test_implicit_column_names2(self):
        query = """
        x = [1 as a, 2 as b];
        y = [from x as x1, x as x2
             emit $2, $3];
        store(y, OUTPUT);"""

        expected_scheme = scheme.Scheme([('a', types.LONG_TYPE),
                                         ('b', types.LONG_TYPE)])
        self.check_result(query, collections.Counter([(1, 2)]),
                          scheme=expected_scheme)

    def test_implicit_column_names3(self):
        query = """
        x = [1 as a, 2 as b];
        y = [from x as x1, x as x2
             emit $2, $1];
        store(y, OUTPUT);"""

        expected_scheme = scheme.Scheme([('a', types.LONG_TYPE),
                                         ('b', types.LONG_TYPE)])
        self.check_result(query, collections.Counter([(1, 2)]),
                          scheme=expected_scheme)

    def test_unbox_index_column_names(self):
        query = """
        x = [1 as a, 2 as b];
        y = [from x as x1, x as x2
             emit x2.$0, x2.$1];
        store(y, OUTPUT);"""

        expected_scheme = scheme.Scheme([('a', types.LONG_TYPE),
                                         ('b', types.LONG_TYPE)])
        self.check_result(query, collections.Counter([(1, 2)]),
                          scheme=expected_scheme)

    def test_duplicate_column_names(self):
        query = """
        x = [1 as a, 2 as b];
        y = [from x as x1, x as x2 emit x1.a, x2.a];
        store(y, OUTPUT);"""

        expected_scheme = scheme.Scheme([('a', types.LONG_TYPE),
                                         ('a1', types.LONG_TYPE)])
        self.check_result(query, collections.Counter([(1, 1)]),
                          scheme=expected_scheme)

    def test_distinct_aggregate_combinations(self):
        """Test to make sure that aggregates of different columns are not
        combined together by the optimizer."""
        query = """
        emp = scan(%s);
        ans = [from emp emit sum(dept_id) as d, sum(salary) as s];
        store(ans, OUTPUT);""" % self.emp_key

        sum_dept_id = sum([e[1] for e in self.emp_table])
        sum_salary = sum([e[3] for e in self.emp_table])
        expected = collections.Counter([(sum_dept_id, sum_salary)])
        self.check_result(query, expected)

    def test_bug_245_dead_code_with_do_while_plan(self):
        """Test to make sure that a dead program (no Stores) with a DoWhile
        throws the correct parse error."""
        with open('examples/deadcode2.myl') as fh:
            query = fh.read()

        with self.assertRaises(MyrialCompileException):
            self.check_result(query, None)

    def test_simple_do_while(self):
        """count to 32 by powers of 2"""
        with open('examples/iteration.myl') as fh:
            query = fh.read()

        expected = collections.Counter([(32, 5)])
        self.check_result(query, expected, output="powersOfTwo")

    def test_pyUDF(self):
        query = """
        T1=scan(%s);
        out = [from T1 emit test(T1.id, T1.dept_id) As ratio];
        store(out, OUTPUT);
        """ % self.emp_key

        self.get_physical_plan(query, udas=[('test', LONG_TYPE)])

    def test_pyUDF_uda(self):
        query = """
        uda Foo(x){
        [0 as _count,0 as _sum];
        [ _count+1, test_uda(_sum, x)];
        [ test_uda(_sum,_count) ];
        };

        T1 = [from scan(%s) as t emit Foo(t.id) As mask];
        store(T1, out);
        """ % self.emp_key

        self.get_physical_plan(query, udas=[('test_uda', LONG_TYPE)])
