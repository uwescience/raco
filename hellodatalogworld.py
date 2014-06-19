from raco.datalog.grammar import parse
from raco.language import MyriaAlgebra
from raco.algebra import LogicalAlgebra
from raco.compile import compile, optimize

query = """
A(x) :- R(x,3)
A(x) :- R(x,y), A(y)
"""

# Now parse it
parsedprogram = parse(query)

exprs = parsedprogram.toRA()
# generate an RA expression
#rules = parsedprogram.rules
#ra = onlyrule.toRA(parsedprogram)

#print exprs
ra = exprs

print query
print ra

print "//-------"


# optimize applies a set of rules to translate a source
# expression to a target expression
#result = optimize(ra, target=PseudoCodeAlgebra, source=LogicalAlgebra)
#result = optimize(ra, target=ProtobufAlgebra, source=LogicalAlgebra)
result = optimize(ra, target=MyriaAlgebra(), source=LogicalAlgebra)

print result

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
