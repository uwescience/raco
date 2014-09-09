from raco import RACompiler

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

    dlog.optimize(target=GrappaAlgebra)

    LOG.info("physical: %s",dlog.physicalplan[0][1])

    # generate code in the target language
    code = ""
    code += comment("Query " + query)
    code += dlog.compile()

    with open(name+'.cpp', 'w') as f:
        f.write(code)


queries = [
("A(s1) :- T1(s1)", "scan"),
("A(s1) :- T1(s1), s1>10", "select"),
("A(s1) :- T1(s1), s1>0, s1<10", "select_conjunction"),
("A(s1,s2) :- T2(s1,s2), s1>10, s2>10", "two_var_select"),
("A(s1,o2) :- T3(s1,p1,o1), R3(o2,p1,o2)", "join"),
("A(a,b,c) :- R2(a,b), S2(b,c)", "two_path"),
("A(a,c) :- R2(a,b), S2(b,c)", "two_hop"),
("A(a,b,c) :- R2(a,b), S2(b,c), T2(c,d)", "three_path"),
("A(a,b,c) :- R2(a,b), S2(b,c), T2(c,a)", "directed_triangles"),
("A(a,b,c,d) :- R2(a,b), S2(b,c), T2(c,d), Z2(d,a)", "directed_squares"),
("A(s1,s2,s3) :- T3(s1,s2,s3), R2(s3,s4), s1<s2, s4<100", "select_then_join"),
("A(a,b) :- R2(a,b), S2(a,b)", "two_match"),
("""A(s1,s2) :- T2(s1,s2)
    A(s1,s2) :- R2(s1,s2)""", "union"),
#("A(a,b,c) :- R(a,b), S(b,c), T(c,a), a<b, b<c", "increasing_triangles"),
#("A(s1,s2,s3) :- T(s1,s2,s3), R(s3,s4), s1<s4", "equi_and_range"),
#("A(s1,s2,s3) :- T(s1,s2),R(s3,s4), s1<s3", "range_join"),
#("A(a,b,c,d,e):-X(a,b),Y(a,c),Z(a,d,e),T(a,b),K(b,a)", "complex_joins"),
]

for q in queries:
    query, name = q
    testEmit(query, 'grappa_'+name)

