
import collections
import math
import unittest

import raco.fakedb
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
import raco.scheme as scheme
import raco.myrial.groupby
import raco.myrial.unpack_from
import raco.myrial.myrial_test as myrial_test

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

    emp_key = "andrew:adhoc:employee"

    dept_table = collections.Counter([
        (1,"accounting",5),
        (2,"human resources",2),
        (3,"engineering",2),
        (4,"sales",7)])

    dept_schema = scheme.Scheme([("id", "int"),
                                 ("name", "string"),
                                 ("manager", "int")])

    dept_key = "andrew:adhoc:department"

    numbers_table = collections.Counter([
        (1, 3),
        (2, 5),
        (3, -2),
        (16, -4.3)])

    numbers_schema = scheme.Scheme([("id", "int"),
                                    ("val", "float")])

    numbers_key = "andrew:adhoc:numbers"

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
        DUMP(emp);
        """ % self.emp_key

        self.run_test(query, self.emp_table)

    def test_scan_dept(self):
        query = """
        dept = SCAN(%s);
        DUMP(dept);
        """ % self.dept_key

        self.run_test(query, self.dept_table)


    def test_bag_comp_emit_star(self):
        query = """
        emp = SCAN(%s);
        bc = [FROM emp EMIT *];
        DUMP(bc);
        """ % self.emp_key

        self.run_test(query, self.emp_table)

    salary_filter_query = """
    emp = SCAN(%s);
    rich = [FROM emp WHERE %s > 25 * 10 * 10 * (5 + 5) EMIT *];
    DUMP(rich);
    """

    salary_expected_result = collections.Counter(
            [x for x in emp_table.elements() if x[3] > 25000])

    def test_bag_comp_filter_large_salary_by_name(self):
        query =  TestQueryFunctions.salary_filter_query % (self.emp_key,
                                                           'salary')
        self.run_test(query, TestQueryFunctions.salary_expected_result)

    def test_bag_comp_filter_large_salary_by_position(self):
        query =  TestQueryFunctions.salary_filter_query % (self.emp_key, '$3')
        self.run_test(query, TestQueryFunctions.salary_expected_result)

    def test_bag_comp_filter_empty_result(self):
        query = """
        emp = SCAN(%s);
        poor = [FROM emp WHERE $3 < (5 * 2) EMIT *];
        DUMP( poor);
        """ % self.emp_key

        expected = collections.Counter()
        self.run_test(query, expected)

    def test_bag_comp_filter_column_compare_ge(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE 2 * $1 >= $0 EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if 2 * x[1] >= x[0]])
        self.run_test(query, expected)

    def test_bag_comp_filter_column_compare_le(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $1 <= 2 * $0 EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[1] <= 2 * x[0]])
        self.run_test(query, expected)

    def test_bag_comp_filter_column_compare_gt(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE 2 * $1 > $0 EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if 2 * x[1] > x[0]])
        self.run_test(query, expected)

    def test_bag_comp_filter_column_compare_lt(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $1 < 2 * $0 EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[1] < 2 * x[0]])
        self.run_test(query, expected)

    def test_bag_comp_filter_column_compare_eq(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $0 * 2 == $1 EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[0] * 2 == x[1]])
        self.run_test(query, expected)

    def test_bag_comp_filter_column_compare_ne(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $0 / $1 != $1 EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[0] / x[1] != x[1]])
        self.run_test(query, expected)

    def test_bag_comp_filter_minus(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $0 + -$1 == $1 EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[0] - x[1] ==  x[1]])
        self.run_test(query, expected)

    def test_bag_comp_filter_and(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE salary == 25000 AND id > dept_id EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000 and
             x[0] > x[1]])
        self.run_test(query, expected)

    def test_bag_comp_filter_or(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE $3 > 25 * 1000 OR id > dept_id EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] > 25000 or
             x[0] > x[1]])
        self.run_test(query, expected)

    def test_bag_comp_filter_not(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE not salary > 25000 EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if not x[3] > 25000])
        self.run_test(query, expected)

    def test_bag_comp_filter_or_and(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE salary == 25000 OR salary == 5000 AND
        dept_id == 1 EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000 or
             (x[3] == 5000 and x[1] == 1)])
        self.run_test(query, expected)

    def test_bag_comp_filter_or_and_not(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE salary == 25000 OR NOT salary == 5000 AND
        dept_id == 1 EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000 or not
             x[3] == 5000 and x[1] == 1])
        self.run_test(query, expected)

    def test_bag_comp_emit_columns(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp WHERE dept_id == 1 EMIT $2, salary=salary];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[2], x[3]) for x in self.emp_table.elements() if x[1] == 1])
        self.run_test(query, expected)

    def test_bag_comp_emit_literal(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp EMIT salary, "bugga bugga"];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[3], "bugga bugga")  for x in self.emp_table.elements()])

        self.run_test(query, expected)

    def test_bag_comp_emit_with_math(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp EMIT salary + 5000, salary - 5000, salary / 5000,
        salary * 5000];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[3] + 5000, x[3] - 5000, x[3] / 5000, x[3] * 5000) \
             for x in self.emp_table.elements()])
        self.run_test(query, expected)

    def test_bag_comp_rename(self):
        query = """
        emp = SCAN(%s);
        out = [FROM emp EMIT name, double_salary=salary * 2];
        out = [FROM out WHERE double_salary > 10000 EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[2], x[3] * 2) for x in self.emp_table.elements() if
             x[3] * 2 > 10000])

        self.run_test(query, expected)

    join_expected = collections.Counter(
        [('Bill Howe', 'human resources'),
         ('Dan Halperin', 'accounting'),
         ('Andrew Whitaker','accounting'),
         ('Shumo Chu', 'human resources'),
         ('Victor Almeida', 'accounting'),
         ('Dan Suciu', 'engineering'),
         ('Magdalena Balazinska', 'accounting')])

    def test_explicit_join(self):
        query = """
        emp = SCAN(%s);
        dept = SCAN(%s);
        out = JOIN(emp, dept_id, dept, id);
        out = [FROM out EMIT emp_name=$2, dept_name=$5];
        DUMP(out);
        """ % (self.emp_key, self.dept_key)

        self.run_test(query, self.join_expected)

    def test_bagcomp_join_via_names(self):
        query = """
        out = [FROM E=SCAN(%s),D=SCAN(%s) WHERE E.dept_id == D.id
              EMIT emp_name=E.name, dept_name=D.name];
        DUMP(out);
        """ % (self.emp_key, self.dept_key)

        self.run_test(query, self.join_expected)

    def test_bagcomp_join_via_pos(self):
        query = """
        E = SCAN(%s);
        D = SCAN(%s);
        out = [FROM E, D WHERE E.$1 == D.$0
              EMIT emp_name=E.name, dept_name=D.$1];
        DUMP(out);
        """ % (self.emp_key, self.dept_key)

        self.run_test(query, self.join_expected)

    # TODO: test with multiple join attributes

    def test_explicit_cross(self):
        query = """
        out = CROSS(SCAN(%s), SCAN(%s));
        DUMP(out);
        """ % (self.emp_key, self.dept_key)

        tuples = [e + d for e in self.emp_table.elements() for
                  d in self.dept_table.elements()]
        expected = collections.Counter(tuples)

        self.run_test(query, expected)

    def test_bagcomp_cross(self):
        query = """
        out = [FROM E=SCAN(%s),D=SCAN(%s) EMIT *];
        DUMP(out);
        """  % (self.emp_key, self.dept_key)

        tuples = [e + d for e in self.emp_table.elements() for
                  d in self.dept_table.elements()]
        expected = collections.Counter(tuples)

        self.run_test(query, expected)

    def test_distinct(self):
        query = """
        out = DISTINCT([FROM X=SCAN(%s) EMIT salary]);
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter([(25000,),(5000,),(90000,)])
        self.run_test(query, expected)

    def test_limit(self):
        query = """
        out = LIMIT(SCAN(%s), 3);
        DUMP(out);
        """ % self.emp_key

        result = self.execute_query(query)
        self.assertEquals(len(result), 3)


    def test_table_literal(self):
        query = """
        X = [FROM Z=["Andrew", salary=(50 * (500 + 500))] EMIT salary];
        DUMP(X);
        """
        expected = collections.Counter([(50000,)])
        self.run_test(query, expected)

    def test_unbox_from_where_single(self):
        query = """
        TH = [25 * 1000];
        emp = SCAN(%s);
        out = [FROM emp WHERE $3 > *TH EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] > 25000])
        self.run_test(query, expected)

    def test_unbox_from_where_multi(self):
        query = """
        TWO = [2];
        FOUR = [4];
        EIGHT = [8];

        emp = SCAN(%s);
        out = [FROM emp WHERE *EIGHT == *TWO**FOUR EMIT *];
        DUMP(out);
        """ % self.emp_key

        self.run_test(query, self.emp_table)

    def test_unbox_from_where_nary_name(self):
        query = """
        CONST = [twenty_five=25, thousand=1000];

        emp = SCAN(%s);
        out = [FROM emp WHERE salary == *CONST.twenty_five *
        *CONST.thousand EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000])

        self.run_test(query, expected)

    def test_unbox_from_where_nary_pos(self):
        query = """
        CONST = [twenty_five=25, thousand=1000];

        emp = SCAN(%s);
        out = [FROM emp WHERE salary == *CONST.$0 *
        *CONST.$1 EMIT *];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [x for x in self.emp_table.elements() if x[3] == 25000])

        self.run_test(query, expected)

    def test_unbox_from_emit_single(self):
        query = """
        THOUSAND = [1000];
        emp = SCAN(%s);
        out = [FROM emp EMIT salary=salary * *THOUSAND];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[3] * 1000,) for x in self.emp_table.elements()])
        self.run_test(query, expected)

    def test_unbox_kitchen_sink(self):
        query = """
        C1 = [a=25, b=100];
        C2 = [a=50, b=1000];

        emp = SCAN(%s);
        out = [FROM emp WHERE salary==*C1.a * *C2.b OR $3==*C1.b * *C2
               EMIT kitchen_sink = dept_id * *C1.b / *C2.a];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter(
            [(x[1] * 2,) for x in self.emp_table.elements() if
             x[3] == 5000 or x[3] == 25000])
        self.run_test(query, expected)


    def __aggregate_expected_result(self, apply_func):
        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[1]].append(t[3])

        tuples = [(key, apply_func(values)) for key, values in
                  result_dict.iteritems()]
        return collections.Counter(tuples)

    def test_max(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT dept_id, MAX(salary)];
        DUMP(out);
        """ % self.emp_key

        self.run_test(query, self.__aggregate_expected_result(max))

    def test_min(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT dept_id, MIN(salary)];
        DUMP(out);
        """ % self.emp_key

        self.run_test(query, self.__aggregate_expected_result(min))

    def test_sum(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT dept_id, SUM(salary)];
        DUMP(out);
        """ % self.emp_key

        self.run_test(query, self.__aggregate_expected_result(sum))

    def test_avg(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT dept_id, AVG(salary)];
        DUMP(out);
        """ % self.emp_key

        def avg(it):
            sum = 0
            cnt = 0
            for val in it:
                sum += val
                cnt += 1
            return sum / cnt

        self.run_test(query, self.__aggregate_expected_result(avg))

    def test_stdev(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT STDEV(salary)];
        DUMP(out);
        """ % self.emp_key

        res = self.execute_query(query)
        tp = res.elements().next()
        self.assertAlmostEqual(tp[0], 34001.8006726)

    def test_count(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT dept_id, COUNT(salary)];
        DUMP(out);
        """ % self.emp_key

        self.run_test(query, self.__aggregate_expected_result(len))

    def test_max_reversed(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT max_salary=MAX(salary), dept_id];
        DUMP(out);
        """ % self.emp_key

        ex = self.__aggregate_expected_result(max)
        ex = collections.Counter([(y,x) for (x,y) in ex])
        self.run_test(query, ex)

    def test_compound_aggregate(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT range=( 2 * (MAX(salary) - MIN(salary))),
        did=dept_id];
        out = [FROM out EMIT dept_id=did, rng=range];
        DUMP(out);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[1]].append(t[3])

        tuples = [(key, 2 * (max(values) - min(values))) for key, values in
                  result_dict.iteritems()]

        expected = collections.Counter(tuples)
        self.run_test(query, expected)

    def test_aggregate_with_unbox(self):
        query = """
        C = [one=1, two=2];
        out = [FROM X=SCAN(%s) EMIT range=MAX(*C.two * salary) -
        MIN( *C.$1 * salary), did=dept_id];
        out = [FROM out EMIT dept_id=did, rng=range];
        DUMP(out);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[1]].append(2 * t[3])

        tuples = [(key, (max(values) - min(values))) for key, values in
                  result_dict.iteritems()]

        expected = collections.Counter(tuples)
        self.run_test(query, expected)

    def test_nary_groupby(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT dept_id, salary, COUNT(name)];
        DUMP(out);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[(t[1], t[3])].append(t[2])

        tuples = [key + (len(values),)
                  for key, values in result_dict.iteritems()]
        expected = collections.Counter(tuples)
        self.run_test(query, expected)

    def test_empty_groupby(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT MAX(salary), COUNT($0), MIN(dept_id*4)];
        DUMP(out);
        """ % self.emp_key

        expected = collections.Counter([(90000, len(self.emp_table), 4)])
        self.run_test(query, expected)

    def test_compound_groupby(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT id+dept_id, COUNT(salary)];
        DUMP(out);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[0] + t[1]].append(t[3])

        tuples = [(key, len(values)) for key, values in result_dict.iteritems()]
        expected = collections.Counter(tuples)
        self.run_test(query, expected)

    def test_impure_aggregate_colref(self):
        """Test of aggregate expression that refers to a grouping column"""
        query = """
        out = [FROM X=SCAN(%s) EMIT
               val=( X.dept_id +  (MAX(X.salary) - MIN(X.salary))),
               did=X.dept_id];

        out = [FROM out EMIT dept_id=did, rng=val];
        DUMP(out);
        """ % self.emp_key

        result_dict = collections.defaultdict(list)
        for t in self.emp_table.elements():
            result_dict[t[1]].append(t[3])

        tuples = [(key, key + (max(values) - min(values))) for key, values in
                  result_dict.iteritems()]

        expected = collections.Counter(tuples)
        self.run_test(query, expected)

    def test_aggregate_illegal_colref(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT
               val=X.dept_id + COUNT(X.salary)];
        DUMP(out);
        """ % self.emp_key

        with self.assertRaises(
                raco.myrial.groupby.InvalidAttributeRefException):
            self.run_test(query, None)

    def test_nested_aggregates_are_illegal(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT id+dept_id, foo=MIN(53 + MAX(salary))];
        DUMP(out);
        """ % self.emp_key

        with self.assertRaises(raco.myrial.groupby.NestedAggregateException):
            self.run_test(query, collections.Counter())

    def test_multiway_bagcomp_with_unbox(self):
        """Return all employees in accounting making less than 30000"""
        query = """
        Salary = [30000];
        Dept = ["accounting"];

        out = [FROM E=SCAN(%s), D=SCAN(%s)
               WHERE E.dept_id == D.id AND D.name == *Dept
               AND E.salary < *Salary EMIT name=E.$2];
        DUMP(out);
        """ % (self.emp_key, self.dept_key)

        expected = collections.Counter([
            ("Andrew Whitaker",),
            ("Victor Almeida",),
            ("Magdalena Balazinska",)])
        self.run_test(query, expected)

    def test_duplicate_bagcomp_aliases_are_illegal(self):
        query = """
        X = SCAN(%s);
        out = [FROM X, X EMIT *];
        DUMP(out);
        """ % (self.emp_key,)

        with self.assertRaises(interpreter.DuplicateAliasException):
            self.run_test(query, collections.Counter())

    def test_bagcomp_column_index_out_of_bounds(self):
        query = """
        E = SCAN(%s);
        D = SCAN(%s);
        out = [FROM E, D WHERE E.$1 == D.$77
              EMIT emp_name=E.name, dept_name=D.$1];
        DUMP(out);
        """ % (self.emp_key, self.dept_key)

        with self.assertRaises(raco.myrial.unpack_from.ColumnIndexOutOfBounds):
            self.run_test(query, collections.Counter())

    def test_abs(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT id, ABS(val)];
        DUMP(out);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a,abs(b)) for a,b in self.numbers_table.elements()])
        self.run_test(query, expected)

    def test_ceil(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT id, CEIL(val)];
        DUMP(out);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a,math.ceil(b)) for a,b in self.numbers_table.elements()])
        self.run_test(query, expected)

    def test_cos(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT id, COS(val)];
        DUMP(out);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a,math.cos(b)) for a,b in self.numbers_table.elements()])
        self.run_test(query, expected)

    def test_floor(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT id, FLOOR(val)];
        DUMP(out);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a,math.floor(b)) for a,b in self.numbers_table.elements()])
        self.run_test(query, expected)

    def test_log(self):
        query = """
        out = [FROM X=SCAN(%s) WHERE val > 0 EMIT id, LOG(val)];
        DUMP(out);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a,math.log(b)) for a,b in self.numbers_table.elements()
             if b > 0])
        self.run_test(query, expected)

    def test_sin(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT id, SIN(val)];
        DUMP(out);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a,math.sin(b)) for a,b in self.numbers_table.elements()])
        self.run_test(query, expected)

    def test_sqrt(self):
        query = """
        out = [FROM X=SCAN(%s) WHERE val >= 0 EMIT id, SQRT(val)];
        DUMP(out);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a,math.sqrt(b)) for a,b in self.numbers_table.elements()
             if b >= 0])
        self.run_test(query, expected)

    def test_tan(self):
        query = """
        out = [FROM X=SCAN(%s) EMIT id, TAN(val)];
        DUMP(out);
        """ % self.numbers_key

        expected = collections.Counter(
            [(a,math.tan(b)) for a,b in self.numbers_table.elements()])
        self.run_test(query, expected)

