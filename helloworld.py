from raco import RACompiler
from raco.language import MyriaAlgebra
from raco.algebra import LogicalAlgebra

# A simple join
query = """
A(x,z) :- R(x,y), S(y,z)
"""

def comment(s):
  print "/*\n%s\n*/" % str(s)

# Create a cmpiler object
dlog = RACompiler()

# parse the query
dlog.fromDatalog(query)
print dlog.logicalplan

dlog.optimize(target=MyriaAlgebra, eliminate_common_subexpressions=False)

# generate code in the target language
code = dlog.compile()
print code

