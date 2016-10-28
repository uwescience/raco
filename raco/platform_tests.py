from abc import ABCMeta, abstractmethod
from collections import Counter


class DatalogPlatformTest(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def check(self, query):
        raise NotImplementedError("{t}.check()".format(t=type(self)))

    @abstractmethod
    def check_file(self, query, name):
        raise NotImplementedError("{t}.check_file()".format(t=type(self)))

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
        self.check(
            "A(a,b,c) :- R2(a,b), S2(b,c), T2(c,a)",
            "directed_triangles"),

    def test_directed_squares(self):
        self.check(
            "A(a,b,c,d) :- R2(a,b), S2(b,c), T2(c,d), R3(d,a,x)",
            "directed_squares"),

    def test_select_then_join(self):
        self.check(
            "A(s1,s2,s3) :- T3(s1,s2,s3), R2(s3,s4), s1<s2, s4<9",
            "select_then_join"),

    def test_unionall(self):
        self.check("""A(s1) :- T1(s1)
    A(s1) :- R1(s1)""", "unionall")

    def test_swap(self):
        self.check("A(y,x) :- R2(x,y)", "swap"),

    def test_apply(self):
        self.check("""A(x,y) :- T2(x,y)
    B(a) :- A(z,a)""", "apply")

    def test_apply_and_self_join(self):
        self.check("""A(x,z) :- T3(x,y,z), y < 4
    B(x,t) :- A(x,z), A(z,t)""", "apply_and_self_join")

    def test_unionall_apply_and_self_join(self):
        self.check("""A(x,y) :- T2(x,y), R1(x), y < 4
            A(x,y) :- R2(x,y), T1(x)
    B(x,z,t) :- A(x,z), A(z,t)""", "unionall_apply_and_self_join")

    def test_unionall_of_join(self):
        self.check("""A(s1,s2) :- T2(s1,s2)
    A(s1,s2) :- R2(s1,s3), T2(s3,s2)""", "unionall_of_join")

    def test_unionall_then_join(self):
        self.check("""A(s1,s2) :- T2(s1,s2)
    A(s1,s2) :- R2(s1,s2)
    B(s1) :- A(s1,s2), S1(s1)""", "unionall_then_join")

    def test_join_of_two_unionalls(self):
        self.check("""A(s1,s2) :- T2(s1,s2)
    A(s1,s2) :- R2(s1,s2)
    B(s1) :- A(s1,s2), A(s1,s3)""", "join_of_two_unionalls")

    def test_join_swap_indexing(self):
        self.check(
            """A(a,h,y) :- T3(a,b,c), R3(x, y, z), S3(g,h,j), z=c, j=x""",
            "join_swap_indexing")

    def test_head_scalar_op(self):
        self.check("""A(a+b) :- R2(a,b)""", "head_scalar_op")

    def test_aggregate_sum(self):
        self.check("""A(SUM(a)) :- R1(a)""", "aggregate_sum")

    def test_aggregate_count(self):
        self.check("""A(COUNT(a)) :- R1(a)""", "aggregate_count")

    def test_aggregate_count_group_one(self):
        self.check(
            """A(b, COUNT(a)) :- R2(a,b)""",
            "aggregate_count_group_one")

    def test_aggregate_count_group_one_notgroup_one(self):
        self.check(
            """A(b, COUNT(a)) :- R3(a,b,c)""",
            "aggregate_count_group_one_notgroup_one")

    def test_aggregate_count_group_one_notgroup_filtered_one(self):
        self.check(
            """A(b, COUNT(a)) :- R3(a,b,c), c<5""",
            "aggregate_count_group_one_notgroup_filtered_one")

    def test_aggregate_of_binop(self):
        self.check("""A(SUM(a+b)) :- R2(a,b)""", "aggregate_of_binop")

    def test_join_of_aggregate_of_join(self):
        self.check("""A(SUM(a), c) :- R2(a,b), T2(b,c)
                      B(x, y) :- A(x, z), S2(z, y)""",
                   "join_of_aggregate_of_join")

    def test_common_index_allowed(self):
        """introduced for #250"""
        self.check(
            """A(a,b,c,d) :- T2(a,b), R2(a,c), R2(a,d)""",
            "common_index_allowed")

    def test_common_index_disallowed(self):
        """introduced for #250"""
        self.check(
            """A(a,b,c,d) :- T2(a,b), R2(a,c), R2(d,a)""",
            "common_index_disallowed")

    def test_file_store(self):
        self.check_file("""store(a,b) :- R2(a,b)""", "store")

    def test_file_fewer_col_store(self):
        self.check_file("""few_col_store(a) :- R2(a,3)""", "few_col_store")

    def test_file_more_col_store(self):
        self.check_file(
            """more_col_store(a,b,c) :- R2(a,b), S2(b,c), T2(c,a)""",
            "more_col_store")

    def test_file_no_tuple_store(self):
        self.check_file("""zero_store(a) :- R2(a,11)""", "zero_store")

import raco.scheme as scheme
import raco.types as types
import raco.myrial.myrial_test as myrial_test


class MyriaLPlatformTestHarness(myrial_test.MyrialTestCase):
    __metaclass__ = ABCMeta

    def setUp(self):
        super(MyriaLPlatformTestHarness, self).setUp()

        self.tables = {}
        for name in ['R', 'S', 'T', 'I', 'D', 'C']:
            for width in [1, 2, 3]:
                tablename = "%s%d" % (name, width)
                fullname = "public:adhoc:%s" % tablename
                self.tables[tablename] = fullname

                if name == 'D':
                    rest_type = types.DOUBLE_TYPE
                elif name == 'C':
                    rest_type = types.STRING_TYPE
                else:
                    rest_type = types.LONG_TYPE

                one = [("a", types.LONG_TYPE)]
                two = one + [("b", rest_type)]
                three = two + [("c", rest_type)]
                # ingest fake data; data is already generated separately for
                # now
                if width == 1:
                    self.db.ingest(fullname, Counter(), scheme.Scheme(one))
                elif width == 2:
                    self.db.ingest(fullname, Counter(), scheme.Scheme(two))
                else:
                    self.db.ingest(fullname, Counter(), scheme.Scheme(three))

    @abstractmethod
    def check(self, query, name):
        pass


class MyriaLPlatformTests(object):

    def myrial_from_sql(self, tables, name):
        code = [
            "{t} = scan({tab});".format(
                t=t,
                tab=self.tables[t]) for t in tables]

        with open("c_test_environment/testqueries/%s.sql" % name) as f:
            code.append("out = %s" % f.read())

        code.append("store(out, OUTPUT);")

        return '\n'.join(code)

    def check_sub_tables(self, query, name, **kwargs):
        self.check(query % self.tables, name, **kwargs)

    def test_scan(self):
        self.check_sub_tables("""
        T1 = SCAN(%(T1)s);
        STORE(T1, OUTPUT);
        """, "scan")

    def test_sink(self):
        """
        Sink still prints on verbose=2, so same as store for testing method
        """
        self.check_sub_tables("""
        T1 = SCAN(%(T1)s);
        SINK(T1);
        """, "scan")

    def test_select(self):
        q = self.myrial_from_sql(['T1'], "select")
        self.check(q, "select")

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

    def test_unionall(self):
        self.check_sub_tables("""
        T1 = SCAN(%(T1)s);
        R1 = SCAN(%(R1)s);
        un = UNIONALL(T1, R1);
        STORE(un, OUTPUT);
        """, "unionall")

    def test_unionall_3(self):
        self.check_sub_tables("""
        T1 = SCAN(%(T1)s);
        R1 = SCAN(%(R1)s);
        S1 = SCAN(%(S1)s);
        un = UNIONALL(T1, R1, S1);
        STORE(un, OUTPUT);
        """, "unionall_3")

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

    def test_idivide(self):
        q = self.myrial_from_sql(['T2'], "idivide").replace("/", "//")
        self.check(q, "idivide")

    def test_apply_and_self_join(self):
        q = self.myrial_from_sql(['T3'], "apply_and_self_join")
        self.check(q, "apply_and_self_join")

    def test_unionall_apply_and_self_join(self):
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
        STORE(out, OUTPUT);""", "unionall_apply_and_self_join")

    def test_unionall_of_join(self):
        self.check_sub_tables("""
        T2 = SCAN(%(T2)s);
        R2 = SCAN(%(R2)s);
        A1 = [FROM T2 EMIT a, b];
        A2 = JOIN(R2, b, T2, a);
        A2b = [FROM A2 EMIT $0, $3];
        out = UNIONALL(A1, A2b);
        STORE(out, OUTPUT);""",
                              "unionall_of_join")

    def test_unionall_then_join(self):
        self.check_sub_tables("""
        T2 = SCAN(%(T2)s);
        R2 = SCAN(%(R2)s);
        S1 = SCAN(%(S1)s);
        A = UNIONALL(T2, R2);
        B = JOIN(A, $0, S1, $0);
        out = [FROM B EMIT $0];
        STORE(out, OUTPUT);
        """, "unionall_then_join")

    def test_join_of_two_unionalls(self):
        self.check_sub_tables("""
        T2 = SCAN(%(T2)s);
        R2 = SCAN(%(R2)s);
        A = UNIONALL(T2, R2);
        B = JOIN(A, $0, A, $0);
        out = [FROM B EMIT $0];
        STORE(out, OUTPUT);
        """, "join_of_two_unionalls")

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

    def test_aggregate_min(self):
        q = self.myrial_from_sql(["T2"], "aggregate_min")
        self.check(q, "aggregate_min")

    def test_aggregate_max(self):
        q = self.myrial_from_sql(["T2"], "aggregate_max")
        self.check(q, "aggregate_max")

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

    def test_join_two_types(self):
        """
        Makes sure that key types are not mixed up.
        A related bug came up when input attribute types were
        inadvertantly computed relative to a hashjoin schema instead of
        the schema of its inputs
        """
        q = self.myrial_from_sql(["C3", "R3"], "join_two_types")
        self.check(q, "join_two_types")

    def test_join_of_two_aggregates(self):
        """
        Goal is to force aggregate result to be insert side of a hash join.
        """
        self.check_sub_tables("""
        D2 = SCAN(%(D2)s);
        D3 = SCAN(%(D3)s);
        agg1 = select a, MIN(b) as mb from D2;
        agg2 = select a, MIN(b) as mb from D3;
        out = select agg1.a, agg2.a
            from agg1, agg2
            where agg1.mb = agg2.mb;
        STORE(out, OUTPUT);
        """, "test_join_of_two_aggregates")

    def test_common_index_allowed(self):
        q = self.myrial_from_sql(["R2", "T2"], "common_index_allowed")
        self.check(q, "common_index_allowed")

    def test_common_index_disallowed(self):
        q = self.myrial_from_sql(["R2", "T2"], "common_index_disallowed")
        self.check(q, "common_index_disallowed")

    def test_matrix_mult(self):
        self.check_sub_tables("""
        T = scan(%(T2)s);
        T1 = T;
        T2 = T;
        MM = [from T1, T2
              where T1.$1 = T2.$0
              emit T1.$0 as src, T2.$1 as dst, count(T1.$0)];
        STORE(MM, OUTPUT);
        """, "matrix_mult")

    def test_two_join_switch(self):
        self.check_sub_tables("""
        R3 = SCAN(%(R3)s);
        S3 = SCAN(%(S3)s);
        T3 = SCAN(%(T3)s);
        J1 = JOIN(R3, $2, S3, $1);
        J2 = JOIN(J1, $3, T3, $0);
        P = [FROM J2 WHERE $0>1 and $3>2 and $6>3 EMIT $0, $8];
        STORE(P, OUTPUT);
        """, "two_join_switch", SwapJoinSides=True)

    def test_select_double(self):
        q = self.myrial_from_sql(["D3"], "select_double")
        self.check(q, "select_double")

    def test_project_string(self):
        q = self.myrial_from_sql(["C3"], "project_string")
        self.check(q, "project_string")

    def test_select_string(self):
        q = self.myrial_from_sql(["C3"], "select_string")
        self.check(q, "select_string")

    def test_join_string_val(self):
        q = self.myrial_from_sql(["C2", "T2"], "join_string_val")
        self.check(q, "join_string_val")

    def test_join_string_key(self):
        q = self.myrial_from_sql(["C3", "C3"], "join_string_key")
        self.check(q, "join_string_key")

    def test_groupby_string_key(self):
        self.check_sub_tables("""
        C2 = SCAN(%(C2)s);
        P = [FROM C2 EMIT SUM($0), $1];
        STORE(P, OUTPUT);
        """, "groupby_string_key")

    def test_groupby_string_multi_key(self):
        self.check_sub_tables("""
        C3 = SCAN(%(C3)s);
        P = [FROM C3 EMIT SUM($0), $1, $2];
        STORE(P, OUTPUT);
        """, "groupby_string_multi_key")

    def test_select_string_literal(self):
        q = self.myrial_from_sql(["C3"], "select_string_literal")
        self.check(q, "select_string_literal")

    def test_aggregate_string(self):
        self.check_sub_tables("""
       C3 = SCAN(%(C3)s);
       P = [FROM C3 EMIT $0, COUNT($1)];
       STORE(P, OUTPUT);
       """, "aggregate_string")

    def test_countstar_string(self):
        self.check_sub_tables("""
       C3 = SCAN(%(C3)s);
       P = [FROM C3 EMIT COUNT($1)];
       STORE(P, OUTPUT);
       """, "countstar_string")

    def test_like_begin_end(self):
        q = self.myrial_from_sql(["C2"], "like_begin_end")
        self.check(q, "like_begin_end")

    def test_like_middle(self):
        q = self.myrial_from_sql(["C2"], "like_middle")
        self.check(q, "like_middle")

    def test_like_end(self):
        q = self.myrial_from_sql(["C2"], "like_end")
        self.check(q, "like_end")

    def test_like_begin(self):
        q = self.myrial_from_sql(["C2"], "like_begin")
        self.check(q, "like_begin")

    def test_aggregate_double(self):
        self.check_sub_tables("""
        D2 = SCAN(%(D2)s);
        out = select a, SUM(b) from D2;
        STORE(out, OUTPUT);
        """, "aggregate_double")

    def test_aggregate_of_binop_double(self):
        self.check_sub_tables("""
        D3 = SCAN(%(D3)s);
        out = select a, MAX(b-c) from D3;
        STORE(out, OUTPUT);
        """, "aggregate_of_binop_double")

    def test_aggregate_of_binop_no_key_unionall_double(self):
        self.check_sub_tables("""
        D3 = SCAN(%(D3)s);
        ma = select MAX(b-c) from D3;
        mi = select MIN(c-b) from D3;
        out = UNIONALL(ma, mi);
        STORE(out, OUTPUT);
        """, "aggregate_of_binop_no_key_unionall_double")

    def test_turn_off_optimizations(self):
        # just testing that it doesn't break anything
        self.check_sub_tables("""
        T1 = SCAN(%(T1)s);
        out = [FROM T1 WHERE a>0 and a<10 EMIT a];
        STORE(out, OUTPUT);
        """, "select_conjunction",
                              no_SplitSelects=True,
                              no_MergeSelects=True,
                              no_PushSelects=True)

    def test_iterator_select(self):
        q = self.myrial_from_sql(['T1'], "select")
        self.check(q, "select", compiler='iterator')

    def test_iterator_apply(self):
        self.check_sub_tables("""
        T2 = SCAN(%(T2)s);
        interm = [FROM T2 EMIT $0, $1];
        out = [FROM interm EMIT $1];
        STORE(out, OUTPUT);
        """, "apply", compiler='iterator')

    def test_iterator_join(self):
        self.check_sub_tables("""
        T3 = SCAN(%(T3)s);
        R3 = SCAN(%(R3)s);
        out = JOIN(T3, b, R3, b);
        out2 = [FROM out WHERE $3 = $5 EMIT $0, $3];
        STORE(out2, OUTPUT);
        """, "join", compiler='iterator')

    def test_iterator_aggregate_sum(self):
        q = self.myrial_from_sql(["R1"], "aggregate_sum")
        self.check(q, "aggregate_sum", compiler='iterator')

    def test_iterator_aggregate_count_group_one(self):
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        out = [FROM R2 EMIT b, COUNT(a)];
        STORE(out, OUTPUT);
        """, "aggregate_count_group_one", compiler='iterator')

    def test_symmetric_hash_join(self):
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        S2 = SCAN(%(S2)s);
        T2 = SCAN(%(T2)s);
        j1 = JOIN(R2, b, S2, a);
        j2 = JOIN(j1, $3, T2, a);
        out = [FROM j2 WHERE $0 = $5 EMIT $0, $1, $3];
        STORE(out, OUTPUT);
        """, "directed_triangles", join_type='symmetric_hash')

    def test_symmetric_hash_join_then_aggregate(self):
        """to fix https://github.com/uwescience/raco/issues/477, caused by
        multiple instances of a pipeline"""
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        S2 = SCAN(%(S2)s);
        T2 = SCAN(%(T2)s);
        j2 = select * from R2, S2, T2 where R2.b=S2.a and S2.a=T2.a;
        a = select SUM($0), $1 from j2;
        STORE(a, OUTPUT);
        """, "join_then_aggregate", join_type='symmetric_hash')

    def test_unionall_then_aggregate(self):
        self.check_sub_tables("""
        R2 = SCAN(%(R2)s);
        S2 = SCAN(%(S2)s);
        u = UNIONALL(R2, S2);
        a = select SUM($0), $1 from u;
        STORE(a, OUTPUT);
        """, "unionall_then_aggregate")

    def test_store_file(self):
        self.check_sub_tables("""
        T1 = SCAN(%(T1)s);
        STORE(T1, OUTPUT);
        """, "scan", emit_print='file')

    def test_q2(self):
        """
        A test resembling sp2bench Q2
        """

        self.check_sub_tables("""
            S = SCAN(R3);
            P = [FROM S T1,
            S T2,
            S T3,
            S T4,
            S T5,
            S T6,
            S T7,
            S T8,
            S T9
WHERE T1.a=T2.a
and T2.a=T3.a
and T3.a=T4.a
and T4.a=T5.a
and T5.a=T6.a
and T6.a=T7.a
and T7.a=T8.a
and T8.a=T9.a
and T1.b = 1 and T1.c > 5
and T2.b = 1
and T3.b = 1
and T4.b = 1
and T5.b = 1
and T6.b = 1
and T7.b = 1
and T8.b = 1
and T9.b = 1
EMIT
T1.a as inproc,
T2.c as author,
T3.c as booktitle,
T4.c as title,
T5.c as proc,
T6.c as ee,
T7.c as page,
T8.c as url,
T9.c as yr];
STORE(P,OUTPUT);""", "q2")
