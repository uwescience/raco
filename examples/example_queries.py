from raco.language import CCAlgebra, GrappaAlgebra
from emitcode import emitCode
import raco.rules as rules
from raco.grappalang import GrappaShuffleHashJoin, GrappaSymmetricHashJoin, GrappaHashJoin

import logging
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

if __name__ == "__main__":
    queries = [
            ("A(s1) :- T1(s1)", "scan"),  # , "select s1 from T1"),
            ("A(s1) :- T1(s1), s1<5", "select"),  # , "select s1 from T1 where s1>10" ),
            ("A(s1) :- T1(s1), s1>0, s1<10", "select_conjunction"),
            ("A(s1,s2) :- T2(s1,s2), s>10, s2>10", "two_var_select"),
            ("A(s1,o2) :- T3(s1,p1,o1), R3(o2,p1,o2)", "join"),
            ("A(a,b) :- R2(a,b), R2(a,c)", "self_join"),
            ("A(a,b,c) :- R2(a,b), S2(b,c)", "two_path"),
            ("A(a,c) :- R2(a,b), S2(b,c)", "two_hop"),
            ("A(a,b,c) :- R2(a,b), S2(b,c), T2(c,d)", "three_path"),
            ("A(a,b,c) :- R2(a,b), R2(b,c), R2(c,d)", "self_three_path"),
            ("A(a,b,c) :- R2(a,b), S2(b,c), T2(c,a)", "directed_triangles"),
            ("A(a,b,c,d) :- R2(a,b), S2(b,c), T2(c,d), Z2(d,a)", "directed_squares"),
            ("A(s1,s2,s3) :- T3(s1,s2,s3), R2(s3,s4), s1<s2, s4<100", "select_then_join"),

            ("""A(s1) :- T1(s1)
    A(s1) :- R1(s1)""", "union"),

            ("A(y,x) :- R2(x,y)", "swap"),

            ("""A(x,y) :- T2(x,y)
    B(a) :- A(z,a)""", "basic_apply"),

            ("""A(x,z) :- T3(x,y,z), y < 4
    B(x,t) :- A(x,z), A(z,t)""", "apply_and_self_join"),

            ("""A(x,y) :- T2(x,y), R1(x), y < 4
            A(x,y) :- R2(x,y), T1(x)
    B(x,z,t) :- A(x,z), A(z,t)""", "union_apply_and_self_join"),
            

            ("""A(s1,s2) :- T2(s1,s2)
    A(s1,s2) :- R2(s1,s3), T2(s3,s2)""", "union_of_join"),

            ("""A(s1,s2) :- T1(s1,s2)
    A(s1,s2) :- R1(s1,s2)
    B(s1) :- A(s1,s2), S1(s1)""", "union_then_join"),

            ("""A(s1,s2) :- T1(s1,s2)
    A(s1,s2) :- R1(s1,s2)
    B(s1) :- A(s1,s2), A(s1,s3)""", "join_of_two_unions"),
            
            ("""A(author) :- R(erdoes, 'rdf:type', 'foaf:Person'),
                            R(doc, 10, erdoes),
                            R(doc, 20, author),
                            erdoes!=author""", "nejoin_forward"),
            ("""A(author) :- R(erdoes, 'rdf:type', 'foaf:Person'),
                            R(doc, 10, erdoes),
                            R(doc, 20, author),
                            author!=erdoes""", "nejoin_rev"),
            

            #("A(a,b) :- R2(a,b), S2(a,b)", "two_match"),

            #("""A(s1,s2) :- R2(s1,s2)
            #    A(s1,s2) :- R2(s1,s3),A(s3,s2)""", "reachable"),
            #("A(a,b,c) :- R(a,b), S(b,c), T(c,a), a<b, b<c", "increasing_triangles"),
            #("A(s1,s2,s3) :- T(s1,s2,s3), R(s3,s4), s1<s4", "equi_and_range"),
            #("A(s1,s2,s3) :- T(s1,s2),R(s3,s4), s1<s3", "range_join"),
            #("A(a,b,c,d,e):-X(a,b),Y(a,c),Z(a,d,e),T(a,b),K(b,a)", "complex_joins"),
            ]

    alg = CCAlgebra
    prefix = ""
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] ==  "grappa":
            alg = GrappaAlgebra
            prefix = "grappa"

    if len(sys.argv) > 2:
        plan = sys.argv[2]

    for q in queries:
        query, name = q
        lst = []
        if prefix: lst.append(prefix)
        if plan: lst.append(plan)
        if name: lst.append(name)
        emitCode(query, "_".join(lst), alg, prefix, plan)

