
# skipping
from nose.tools import nottest


class PlatformTest(object):

            # @nottest: excluding these tests from nosetests
    # Currently running clang_tests in the hosted travis.ci environment fails
    # to run the compiled C++ programs. Exit code -4 and no output on stderr/stdout.
    # Run these tests separately with `python clang_tests.py` from the root of datalogcompiler/
    @nottest
    def test_scan(self):
        self.check("A(s1) :- T1(s1)", "scan")

    @nottest
    def test_select(self):
        self.check("A(s1) :- T1(s1), s1>5", "select")

    @nottest
    def test_join(self):
        self.check("A(s1,o2) :- T3(s1,p1,o1), R3(o2,p1,o2)", "join")

    @nottest
    def test_select_conjunction(self):
        self.check("A(s1) :- T1(s1), s1>0, s1<10", "select_conjunction")

    @nottest
    def test_two_var_select(self):
        self.check("A(s1,s2) :- T2(s1,s2), s1<9, s2<9", "two_var_select")

    @nottest
    def test_self_join(self):
        self.check("A(a,b) :- R2(a,b), R2(a,c)", "self_join")

    @nottest
    def test_two_path(self):
        self.check("A(a,b,c) :- R2(a,b), S2(b,c)", "two_path")

    @nottest
    def test_two_hop(self):
        self.check("A(a,c) :- R2(a,b), S2(b,c)", "two_hop")

    @nottest
    def test_three_path(self):
        self.check("A(a,b,c) :- R2(a,b), S2(b,c), T2(c,d)", "three_path")

    @nottest
    def test_self_three_path(self):
        self.check("A(a,b,c) :- R2(a,b), R2(b,c), R2(c,d)", "self_three_path"),

    @nottest
    def test_directed_triangles(self):
        self.check("A(a,b,c) :- R2(a,b), S2(b,c), T2(c,a)", "directed_triangles"),

    @nottest
    def test_directed_squares(self):
        self.check("A(a,b,c,d) :- R2(a,b), S2(b,c), T2(c,d), R3(d,a,x)", "directed_squares"),

    @nottest
    def test_select_then_join(self):
        self.check("A(s1,s2,s3) :- T3(s1,s2,s3), R2(s3,s4), s1<s2, s4<9", "select_then_join"),

    # TODO: All unions are currently treated as unionAll
    @nottest
    def test_union(self):
        self.check("""A(s1) :- T1(s1)
    A(s1) :- R1(s1)""", "union")

    @nottest
    def test_swap(self):
        self.check("A(y,x) :- R2(x,y)", "swap"),

    @nottest
    def test_apply(self):
        self.check("""A(x,y) :- T2(x,y)
    B(a) :- A(z,a)""", "apply")

    @nottest
    def test_apply_and_self_join(self):
        self.check("""A(x,z) :- T3(x,y,z), y < 4
    B(x,t) :- A(x,z), A(z,t)""", "apply_and_self_join")

    @nottest
    def test_union_apply_and_self_join(self):
        self.check("""A(x,y) :- T2(x,y), R1(x), y < 4
            A(x,y) :- R2(x,y), T1(x)
    B(x,z,t) :- A(x,z), A(z,t)""", "union_apply_and_self_join")

    @nottest
    def test_union_of_join(self):
        self.check("""A(s1,s2) :- T2(s1,s2)
    A(s1,s2) :- R2(s1,s3), T2(s3,s2)""", "union_of_join")

    @nottest
    def test_union_then_join(self):
        self.check("""A(s1,s2) :- T2(s1,s2)
    A(s1,s2) :- R2(s1,s2)
    B(s1) :- A(s1,s2), S1(s1)""", "union_then_join")

    @nottest
    def test_join_of_two_unions(self):
        self.check("""A(s1,s2) :- T2(s1,s2)
    A(s1,s2) :- R2(s1,s2)
    B(s1) :- A(s1,s2), A(s1,s3)""", "join_of_two_unions")

    @nottest
    def test_join_swap_indexing(self):
        self.check("""A(a,h,y) :- T3(a,b,c), R3(x, y, z), S3(g,h,j), z=c, j=x""", "join_swap_indexing")

    @nottest
    def test_head_scalar_op(self):
        self.check("""A(a+b) :- R2(a,b)""", "head_scalar_op")

    @nottest
    def test_aggregate_sum(self):
        self.check("""A(SUM(a)) :- R1(a)""", "aggregate_sum")

    @nottest
    def test_aggregate_count(self):
        self.check("""A(COUNT(a)) :- R1(a)""", "aggregate_count")

    @nottest
    def test_aggregate_count_group_one(self):
        self.check("""A(b, COUNT(a)) :- R2(a,b)""", "aggregate_count_group_one")

    @nottest
    def test_aggregate_count_group_one_notgroup_one(self):
        self.check("""A(b, COUNT(a)) :- R3(a,b,c)""", "aggregate_count_group_one_notgroup_one")

    @nottest
    def test_aggregate_count_group_one_notgroup_filtered_one(self):
        self.check("""A(b, COUNT(a)) :- R3(a,b,c), c<5""", "aggregate_count_group_one_notgroup_filtered_one")

    @nottest
    def test_aggregate_of_binop(self):
        self.check("""A(SUM(a+b)) :- R2(a,b)""", "aggregate_of_binop")

    @nottest
    def test_join_of_aggregate_of_join(self):
        self.check("""A(SUM(a), c) :- R2(a,b), T2(b,c)
                      B(x, y) :- A(x, z), S2(z, y)""", "join_of_aggregate_of_join")

    @nottest
    def test_common_index_allowed(self):
        """introduced for #250"""
        self.check("""A(a,b,c,d) :- T2(a,b), R2(a,c), R2(a,d)""", "common_index_allowed")
    
    @nottest
    def test_common_index_disallowed(self):
        """introduced for #250"""
        self.check("""A(a,b,c,d) :- T2(a,b), R2(a,c), R2(d,a)""", "common_index_disallowed")
