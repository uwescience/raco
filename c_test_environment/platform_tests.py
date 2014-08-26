class DatalogPlatformTest(object):

    # Run these tests manually with
    #    `python c_test_environment/clang_datalog_tests.py` from `raco/`
    #
    # Currently Grappa tests are disabled unless invoked manually.

    def test_scan(self):
        self.check("A(s1) :- T1(s1)", "scan")

    def test_select(self):
        self.check("A(s1) :- T1(s1), s1>5", "select")

    def test_join(self):
        self.check("A(s1,o2) :- T3(s1,p1,o1), R3(o2,p1,o2)", "join")

    def test_select_conjunction(self):
        self.check("A(s1) :- T1(s1), s1>0, s1<10", "select_conjunction")

    def test_two_var_select(self):
        self.check("A(s1,s2) :- T2(s1,s2), s1<9, s2<9", "two_var_select")

    def test_self_join(self):
        self.check("A(a,b) :- R2(a,b), R2(a,c)", "self_join")

    def test_two_path(self):
        self.check("A(a,b,c) :- R2(a,b), S2(b,c)", "two_path")

    def test_two_hop(self):
        self.check("A(a,c) :- R2(a,b), S2(b,c)", "two_hop")

    def test_three_path(self):
        self.check("A(a,b,c) :- R2(a,b), S2(b,c), T2(c,d)", "three_path")

    def test_self_three_path(self):
        self.check("A(a,b,c) :- R2(a,b), R2(b,c), R2(c,d)", "self_three_path"),

    def test_directed_triangles(self):
        self.check("A(a,b,c) :- R2(a,b), S2(b,c), T2(c,a)", "directed_triangles"),

    def test_directed_squares(self):
        self.check("A(a,b,c,d) :- R2(a,b), S2(b,c), T2(c,d), R3(d,a,x)", "directed_squares"),

    def test_select_then_join(self):
        self.check("A(s1,s2,s3) :- T3(s1,s2,s3), R2(s3,s4), s1<s2, s4<9", "select_then_join"),

    # TODO: All unions are currently treated as unionAll
    def test_union(self):
        self.check("""A(s1) :- T1(s1)
    A(s1) :- R1(s1)""", "union")

    def test_swap(self):
        self.check("A(y,x) :- R2(x,y)", "swap"),

    def test_apply(self):
        self.check("""A(x,y) :- T2(x,y)
    B(a) :- A(z,a)""", "apply")

    def test_apply_and_self_join(self):
        self.check("""A(x,z) :- T3(x,y,z), y < 4
    B(x,t) :- A(x,z), A(z,t)""", "apply_and_self_join")

    def test_union_apply_and_self_join(self):
        self.check("""A(x,y) :- T2(x,y), R1(x), y < 4
            A(x,y) :- R2(x,y), T1(x)
    B(x,z,t) :- A(x,z), A(z,t)""", "union_apply_and_self_join")

    def test_union_of_join(self):
        self.check("""A(s1,s2) :- T2(s1,s2)
    A(s1,s2) :- R2(s1,s3), T2(s3,s2)""", "union_of_join")

    def test_union_then_join(self):
        self.check("""A(s1,s2) :- T2(s1,s2)
    A(s1,s2) :- R2(s1,s2)
    B(s1) :- A(s1,s2), S1(s1)""", "union_then_join")

    def test_join_of_two_unions(self):
        self.check("""A(s1,s2) :- T2(s1,s2)
    A(s1,s2) :- R2(s1,s2)
    B(s1) :- A(s1,s2), A(s1,s3)""", "join_of_two_unions")

    def test_join_swap_indexing(self):
        self.check("""A(a,h,y) :- T3(a,b,c), R3(x, y, z), S3(g,h,j), z=c, j=x""", "join_swap_indexing")

    def test_head_scalar_op(self):
        self.check("""A(a+b) :- R2(a,b)""", "head_scalar_op")

    def test_aggregate_sum(self):
        self.check("""A(SUM(a)) :- R1(a)""", "aggregate_sum")

    def test_aggregate_count(self):
        self.check("""A(COUNT(a)) :- R1(a)""", "aggregate_count")

    def test_aggregate_count_group_one(self):
        self.check("""A(b, COUNT(a)) :- R2(a,b)""", "aggregate_count_group_one")

    def test_aggregate_count_group_one_notgroup_one(self):
        self.check("""A(b, COUNT(a)) :- R3(a,b,c)""", "aggregate_count_group_one_notgroup_one")

    def test_aggregate_count_group_one_notgroup_filtered_one(self):
        self.check("""A(b, COUNT(a)) :- R3(a,b,c), c<5""", "aggregate_count_group_one_notgroup_filtered_one")

    def test_aggregate_of_binop(self):
        self.check("""A(SUM(a+b)) :- R2(a,b)""", "aggregate_of_binop")

    def test_join_of_aggregate_of_join(self):
        self.check("""A(SUM(a), c) :- R2(a,b), T2(b,c)
                      B(x, y) :- A(x, z), S2(z, y)""", "join_of_aggregate_of_join")

    def test_common_index_allowed(self):
        """introduced for #250"""
        self.check("""A(a,b,c,d) :- T2(a,b), R2(a,c), R2(a,d)""", "common_index_allowed")
    
    def test_common_index_disallowed(self):
        """introduced for #250"""
        self.check("""A(a,b,c,d) :- T2(a,b), R2(a,c), R2(d,a)""", "common_index_disallowed")

    def test_file_store(self):
        self.check_file("""store(a,b) :- R2(a,b)""", "store")

    def test_file_fewer_col_store(self):
        self.check_file("""few_col_store(a) :- R2(a,3)""", "few_col_store")

    def test_file_more_col_store(self):
        self.check_file("""more_col_store(a,b,c) :- R2(a,b), S2(b,c), T2(c,a)""", "more_col_store")

    def test_file_no_tuple_store(self):
        self.check_file("""zero_store(a) :- R2(a,11)""", "zero_store")

import raco.scheme as scheme
import raco.types as types
import raco.myrial.myrial_test as myrial_test


class MyriaLPlatformTestHarness(myrial_test.MyrialTestCase):

    def setUp(self):
        super(MyriaLPlatformTestHarness, self).setUp()

        self.tables = {}
        for name in ['R', 'S', 'T']:
            for width in [1, 2, 3]:
                tablename = "%s%d" % (name, width)
                fullname = "public:adhoc:%s" % tablename
                self.tables[tablename] = fullname

                one = [("a", types.INT_TYPE)]
                two = one + [("b", types.INT_TYPE)]
                three = two + [("c", types.INT_TYPE)]
                # ingest fake data; data is already generated separately for now
                if width == 1:
                    self.db.ingest(fullname, [], scheme.Scheme(one))
                elif width == 2:
                    self.db.ingest(fullname, [], scheme.Scheme(two))
                else:
                    self.db.ingest(fullname, [], scheme.Scheme(three))


class MyriaLPlatformTests(object):

    def myrial_from_sql(self, tables, name):
        code = ""
        for t in tables:
           code += "%s = SCAN(%s);" % (t, self.tables[t])

        with open("c_test_environment/testqueries/%s.sql" % name) as f:
            code += "out = %s" % f.read()

        code += "STORE(out, OUTPUT);"

        return code

    def check_sub_tables(self, query, name):
        self.check(query % self.tables, name)

    def test_scan(self):
        self.check_sub_tables("""
        T1 = SCAN(%(T1)s);
        STORE(T1, OUTPUT);
        """, "scan")

    def test_select(self):
        self.check_sub_tables("""
        T1 = SCAN(%(T1)s);
        x = [FROM T1 WHERE a>5 EMIT a];
        STORE(x, OUTPUT);
        """, "select")

    def test_join(self):
        self.check_sub_tables("""
        T3 = SCAN(%(T3)s);
        R3 = SCAN(%(R3)s);
        out = JOIN(T3, b, R3, b);
        out2 = [FROM out WHERE $3 = $5 EMIT $0, $3];
        STORE(out2, OUTPUT);
        """, "join")

    def test_select_conjunction(self):
        self.check_sub_tables("""
        T1 = SCAN(%(T1)s);
        out = [FROM T1 WHERE a>0 and a<10 EMIT a];
        STORE(out, OUTPUT);
        """, "select_conjunction")

    def test_two_var_select(self):
        self.check_sub_tables("""
        T2 = SCAN(%(T2)s);
        out = [FROM T2 WHERE a<9 and b<9 EMIT a,b];
        STORE(out, OUTPUT);
        """, "two_var_select")

    def test_self_join(self):
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        j = JOIN(R2, a, R2, a);
        out = [FROM j EMIT $0, $1];
        STORE(out, OUTPUT);
        """, "self_join")

    def test_two_path(self):
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        S2 = SCAN(%(S2)s);
        j = JOIN(R2, b, S2, a);
        out = [FROM j EMIT $0, $1, $3];
        STORE(out, OUTPUT);
        """, "two_path")

    def test_two_hop(self):
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        S2 = SCAN(%(S2)s);
        j = JOIN(R2, b, S2, a);
        out = [FROM j EMIT $0, $3];
        STORE(out, OUTPUT);
        """, "two_hop")

    def test_three_path(self):
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        S2 = SCAN(%(S2)s);
        T2 = SCAN(%(T2)s);
        j1 = JOIN(R2, b, S2, a);
        j2 = JOIN(j1, $3, T2, a);
        out = [FROM j2 EMIT $0, $1, $3];
        STORE(out, OUTPUT);
        """, "three_path")

    def test_self_three_path(self):
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        j1 = JOIN(R2, b, R2, a);
        j2 = JOIN(j1, $3, R2, a);
        out = [FROM j2 EMIT $0, $1, $3];
        STORE(out, OUTPUT);
        """, "self_three_path")

    def test_directed_triangles(self):
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        S2 = SCAN(%(S2)s);
        T2 = SCAN(%(T2)s);
        j1 = JOIN(R2, b, S2, a);
        j2 = JOIN(j1, $3, T2, a);
        out = [FROM j2 WHERE $0 = $5 EMIT $0, $1, $3];
        STORE(out, OUTPUT);
        """, "directed_triangles")

    def test_directed_squares(self):
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        S2 = SCAN(%(S2)s);
        T2 = SCAN(%(T2)s);
        R3 = SCAN(%(R3)s);
        out = select R2.$0 as a, S2.$0 as b, T2.$0 as c, R3.$0 as d
              from R2, S2, T2, R3
              where R2.$1 = S2.$0 and S2.$1 = T2.$0 and T2.$1 = R3.$0
                and R3.$1 = R2.$0;
        STORE(out, OUTPUT);
        """, "directed_squares")

    def test_select_then_join(self):
        self.check_sub_tables("""
        T3 = SCAN(%(T3)s);
        R2 = SCAN(%(R2)s);
        j = JOIN(T3, c, R2, a);
        out = [FROM j WHERE $0 < $1 and $4 < 9 EMIT $0,$1,$2];
        STORE(out, OUTPUT);
        """, "select_then_join")

    # TODO: All unions are currently treated as unionAll
    def test_union(self):
        self.check_sub_tables("""
        T1 = SCAN(%(T1)s);
        R1 = SCAN(%(R1)s);
        un = UNIONALL(T1, R1);
        STORE(un, OUTPUT);
        """, "union")

    def test_swap(self):
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        out = [FROM R2 EMIT b, a];
        STORE(out, OUTPUT);
        """, "swap")

    def test_apply(self):
        self.check_sub_tables("""
        T2 = SCAN(%(T2)s);
        interm = [FROM T2 EMIT $0, $1];
        out = [FROM interm EMIT $1];
        STORE(out, OUTPUT);
        """, "apply")

    def test_apply_and_self_join(self):
        q = self.myrial_from_sql(['T3'], "apply_and_self_join")
        self.check(q, "apply_and_self_join")

    def test_union_apply_and_self_join(self):
        self.check_sub_tables("""
        T2 = SCAN(%(T2)s);
        R1 = SCAN(%(R1)s);
        R2 = SCAN(%(R2)s);
        T1 = SCAN(%(T1)s);
        Aj1 = JOIN(T2, a, R1, a);
        A1 = [FROM Aj1 WHERE $1 < 4 EMIT $0 as x, $1 as y];
        Aj2 = JOIN(R2, a, T1, a);
        A2 = [FROM Aj2 EMIT $0 as x, $1 as y];
        AU = UNIONALL(A1, A2);
        B = JOIN(AU, $1, AU, $0);
        out = [FROM B EMIT $0 as x, $1 as y, $3 as t];
        STORE(out, OUTPUT);""", "union_apply_and_self_join")

    def test_union_of_join(self):
        self.check_sub_tables("""
        T2 = SCAN(%(T2)s);
        R2 = SCAN(%(R2)s);
        A1 = [FROM T2 EMIT a, b];
        A2 = JOIN(R2, b, T2, a);
        A2b = [FROM A2 EMIT $0, $3];
        out = UNIONALL(A1, A2b);
        STORE(out, OUTPUT);""",
                              "union_of_join")

    def test_union_then_join(self):
        self.check_sub_tables("""
        T2 = SCAN(%(T2)s);
        R2 = SCAN(%(R2)s);
        S1 = SCAN(%(S1)s);
        A = UNIONALL(T2, R2);
        B = JOIN(A, $0, S1, $0);
        out = [FROM B EMIT $0];
        STORE(out, OUTPUT);
        """, "union_then_join")

    def test_join_of_two_unions(self):
        self.check_sub_tables("""
        T2 = SCAN(%(T2)s);
        R2 = SCAN(%(R2)s);
        A = UNIONALL(T2, R2);
        B = JOIN(A, $0, A, $0);
        out = [FROM B EMIT $0];
        STORE(out, OUTPUT);
        """, "join_of_two_unions")

    def test_join_swap_indexing(self):
        q = self.myrial_from_sql(["T3", "S3", "R3"], "join_swap_indexing")
        self.check(q, "join_swap_indexing")

    def test_head_scalar_op(self):
        q = self.myrial_from_sql(["R2"], "head_scalar_op")
        self.check(q, "head_scalar_op")

    def test_aggregate_sum(self):
        q = self.myrial_from_sql(["R1"], "aggregate_sum")
        self.check(q, "aggregate_sum")

    def test_aggregate_count(self):
        q = self.myrial_from_sql(["R1"], "aggregate_count")
        self.check(q, "aggregate_count")

    def test_aggregate_count_group_one(self):
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        out = [FROM R2 EMIT b, COUNT(a)];
        STORE(out, OUTPUT);
        """, "aggregate_count_group_one")

    def test_aggregate_count_group_one_notgroup_one(self):
        self.check_sub_tables("""
        R3 = SCAN(%(R3)s);
        out = [FROM R3 EMIT b, COUNT(a)];
        STORE(out, OUTPUT);
        """, "aggregate_count_group_one_notgroup_one")

    def test_aggregate_count_group_one_notgroup_filtered_one(self):
        self.check_sub_tables("""
        R3 = SCAN(%(R3)s);
        out = [FROM R3 WHERE R3.c < 5 EMIT b, COUNT(a)];
        STORE(out, OUTPUT);
        """, "aggregate_count_group_one_notgroup_filtered_one")

    def test_aggregate_of_binop(self):
        q = self.myrial_from_sql(["R2"], "aggregate_of_binop")
        self.check(q, "aggregate_of_binop")

    def test_join_of_aggregate_of_join(self):
        self.check_sub_tables("""
        T2 = SCAN(%(T2)s);
        R2 = SCAN(%(R2)s);
        S2 = SCAN(%(S2)s);
        Ap = JOIN(R2, b, T2, a);
        A = [FROM Ap EMIT SUM($0), $3];
        B = JOIN(A, $1, S2, a);
        out = [FROM B EMIT $0, $3];
        STORE(out, OUTPUT);
        """, "join_of_aggregate_of_join")

    def test_common_index_allowed(self):
        q = self.myrial_from_sql(["R2", "T2"], "common_index_allowed")
        self.check(q, "common_index_allowed")

    def test_common_index_disallowed(self):
        q = self.myrial_from_sql(["R2", "T2"], "common_index_disallowed")
        self.check(q, "common_index_disallowed")
