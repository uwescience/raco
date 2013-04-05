from raco import RACompiler
from raco.language import CCAlgebra, MyriaAlgebra
from raco.algebra import LogicalAlgebra

# A simple join
query = "A(s1,o2) :- T(s1,p1,o), R(o,p2,o2)"

#"""
#A(s1,o2) :- R(s1,p1,o1), T(o1,p2,o2)
#"""

def comment(s):
  print "/*\n%s\n*/" % str(s)

# Create a compiler object
dlog = RACompiler()

# parse the query
dlog.fromDatalog(query)
#print dlog.parsed
#print dlog.logicalplan

dlog.optimize(target=CCAlgebra, eliminate_common_subexpressions=False)

#print dlog.physicalplan

# generate code in the target language
code = dlog.compile()
print code
