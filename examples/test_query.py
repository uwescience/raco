from raco import RACompiler
from raco.language import CCAlgebra, MyriaAlgebra
from raco.algebra import LogicalAlgebra

import logging
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def testEmit(query, name):
    LOG.info("compiling %s", query)

    # Create a compiler object
    dlog = RACompiler()

    # parse the query
    dlog.fromDatalog(query)
    #print dlog.parsed
    LOG.info("logical: %s",dlog.logicalplan)

    dlog.optimize(target=CCAlgebra, eliminate_common_subexpressions=False)

    LOG.info("physical: %s",dlog.physicalplan[0][1])

    # generate code in the target language
    code = dlog.compile()
    
    with open(name+'.c', 'w') as f:
        f.write(code)


queries = [
("A(s1) :- T(s1)", "scan"),
("A(s1) :- T(s1), s>10", "select"),
("A(s1) :- T(s1), s>0, s<10", "select_conjunction"),
("A(s1,s2) :- T(s1,s2), s>10, s2>10", "two_var_select"),
("A(s1,o2) :- T(s1,p1,o1), R(o2,p1,o2)", "join"),
("A(a,b,c) :- R(a,b), S(b,c)", "two_path"),
("A(a,c) :- R(a,b), S(b,c)", "two_hop"),
("A(a,b,c) :- R(a,b), S(b,c), T(c,d)", "three_path"),
("A(a,b,c) :- R(a,b), S(b,c), T(c,a)", "directed_triangles"),
("A(s1,s2,s3) :- T(s1,s2,s3), R(s3,s4), s1<s2, s4<100", "select_then_join"),
#("A(a,b,c) :- R(a,b), S(b,c), T(c,a), a<b, b<c", "increasing_triangles"),
#("A(s1,s2,s3) :- T(s1,s2,s3), R(s3,s4), s1<s4", "equi_and_range"),
#("A(s1,s2,s3) :- T(s1,s2),R(s3,s4), s1<s3", "range_join"),
#("A(a,b,c,d,e):-X(a,b),Y(a,c),Z(a,d,e),T(a,b),K(b,a)", "complex_joins"),
]

for q in queries:
    query, name = q
    testEmit(query, name)

