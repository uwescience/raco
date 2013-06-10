from raco import RACompiler
from raco.language import MyriaAlgebra
from raco.algebra import LogicalAlgebra

# A simple join
simple_query = """
A(x,z) :- R(x,y), S(y,z)
"""
# A multi-join version
multi_query = """
A(x,w) :- R(x,y,z), S(y,z,w)
"""

# Which one do we use?
query = simple_query

def comment(s):
  print "/*\n%s\n*/" % str(s)

# Create a cmpiler object
dlog = RACompiler()

# parse the query
dlog.fromDatalog(query)
print "************ LOGICAL PLAN *************"
print dlog.logicalplan
print

# Optimize the query, includes producing a physical plan
print "************ PHYSICAL PLAN *************"
dlog.optimize(target=MyriaAlgebra, eliminate_common_subexpressions=False)
print dlog.physicalplan
print

# generate code in the target language
code = dlog.compile()
print "************ CODE *************"
print code
print
