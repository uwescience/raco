from raco.datalog.grammar import parse
from raco.scheme import Scheme
from raco.catalog import ASCIIFile
from raco.language import PythonAlgebra, PseudoCodeAlgebra, CCAlgebra#, ProtobufAlgebra
from raco.algebra import LogicalAlgebra
from raco.compile import compile, optimize, common_subexpression_elimination, showids

query = """
A(x,z) :- R(x,p1,y,c1), T(y,p2,z,c2), z=217772631
"""

# Now parse it
parsedprogram = parse(query)

exprs = parsedprogram.toRA()
# generate an RA expression
#rules = parsedprogram.rules
#ra = onlyrule.toRA(parsedprogram)

#print exprs
ra = exprs

#print query
#comment(ra)


print "//-------"

# optimize applies a set of rules to translate a source 
# expression to a target expression
#result = optimize(ra, target=PseudoCodeAlgebra, source=LogicalAlgebra)
#result = optimize(ra, target=ProtobufAlgebra, source=LogicalAlgebra)
result = optimize(ra, target=CCAlgebra, source=LogicalAlgebra)

#comment(result)
physicalplan = result
#physicalplan = common_subexpression_elimination(result)
#comment(physicalplan)
#for x in showids(physicalplan):
#  print x

# generate code in the target language
#print compile(physicalplan)
#compile(physicalplan)
print compile(physicalplan)
