from raco import RACompiler
from raco.language import CCAlgebra, MyriaAlgebra
from raco.algebra import LogicalAlgebra

import logging
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

def comment(s):
  return "/*\n%s\n*/\n" % str(s)

def testEmit(query, name):
    LOG.info("compiling %s: %s", name, query)

    # Create a compiler object
    dlog = RACompiler()

    # parse the query
    dlog.fromDatalog(query)
    #print dlog.parsed
    LOG.info("logical: %s",dlog.logicalplan)

    dlog.optimize(target=CCAlgebra, eliminate_common_subexpressions=False)

    LOG.info("physical: %s",dlog.physicalplan[0][1])

    # generate code in the target language
    code = ""
    code += comment("Query " + query)
    code += dlog.compile()
    
    with open(name+'.cpp', 'w') as f:
        f.write(code)


queries = [
("A(s1) :- T1(s1)", "scan"),#, "select s1 from T1"), 
("A(s1) :- T1(s1), s1>10", "select"),#, "select s1 from T1 where s1>10" ),
("A(s1) :- T1(s1), s1>0, s1<10", "select_conjunction"),
("A(s1,s2) :- T2(s1,s2), s>10, s2>10", "two_var_select"),
("A(s1,o2) :- T3(s1,p1,o1), R3(o2,p1,o2)", "join"),
("A(a,b,c) :- R2(a,b), S2(b,c)", "two_path"),
("A(a,c) :- R2(a,b), S2(b,c)", "two_hop"),
("A(a,b,c) :- R2(a,b), S2(b,c), T2(c,d)", "three_path"),
("A(a,b,c) :- R2(a,b), S2(b,c), T2(c,a)", "directed_triangles"),
("A(s1,s2,s3) :- T3(s1,s2,s3), R2(s3,s4), s1<s2, s4<100", "select_then_join"),
("Q3a(article) :- sp2bench_1m(article, 'rdf:type', 'bench:Article'), sp2bench_1m(article, 'swrc:pages', value)", "sp2_Q3a"),
("Q1(yr) :- sp2bench_1m(journal, 'rdf:type', 'bench:Journal'), sp2bench_1m(journal, 'dc:title', 'Journal 1 (1940)'), sp2bench_1m(journal, 'dcterms:issued', yr)", "sp2_Q1"),

("""A(s1) :- T1(s1)
    A(s1) :- R1(s1)""", "union"),

("A(y,x) :- R2(x,y)", "swap"),

("""A(x,y) :- T2(x,y)
    B(a) :- A(z,a)""", "basic_apply"),

("""A(x,z) :- T3(x,y,z), y < 4
    B(x,t) :- A(x,z), A(z,t)""", "apply_and_self_join"),

("""A(s1,s2) :- T2(s1,s2)
    A(s1,s2) :- R2(s1,s3), T2(s3,s2)""", "union_of_join"),

("""A(s1,s2) :- T1(s1,s2)
    A(s1,s2) :- R1(s1,s2)
    B(s1) :- A(s1,s2), S1(s1)""", "union_then_join"),

#("""A(s1,s2) :- R2(s1,s2)
#    A(s1,s2) :- R2(s1,s3),A(s3,s2)""", "reachable"),
#("A(a,b,c) :- R(a,b), S(b,c), T(c,a), a<b, b<c", "increasing_triangles"),
#("A(s1,s2,s3) :- T(s1,s2,s3), R(s3,s4), s1<s4", "equi_and_range"),
#("A(s1,s2,s3) :- T(s1,s2),R(s3,s4), s1<s3", "range_join"),
#("A(a,b,c,d,e):-X(a,b),Y(a,c),Z(a,d,e),T(a,b),K(b,a)", "complex_joins"),
]

for q in queries:
    query, name = q
    testEmit(query, name)

