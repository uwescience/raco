from raco.datalog.grammar import parse
from raco.scheme import Scheme
from raco.catalog import ASCIIFile
from raco.language import PythonAlgebra, PseudoCodeAlgebra, CCAlgebra, ProtobufAlgebra
from raco.algebra import LogicalAlgebra
from raco.compile import compile, optimize

# declare the schema for each relation
sch = Scheme([("subject", int), ("predicate", int), ("object", int)])

# Create a relation object.  We can add formats here as needed.
R = ASCIIFile("R", sch)

# Now write the Datalog expression
query = """
A(x,z) :- R(x,p1,y,t1),R(y,p2,z,t2),R(z,p3,w,t3)
"""
query = """A(x,z) :- R(x,y),S(y,z,a,b,c,d),T(c,x)"""
query = """A(x,y) :- S(x), R(y,z), T(z,a)"""
query = """A(x) :- R(x,y),S(y,z,4),z<3"""
#query = """A(x) :- S(y,z,4)"""
query = """A(x,z) :- R(x,y),S(y,z),T(z,x)"""

#print query
# Now parse it
parsedprogram = parse(query)

# generate an RA expression
onlyrule = parsedprogram.rules[0]
ra = onlyrule.toRA(parsedprogram)

print query
print ra

# optimize applies a set of rules to translate a source 
# expression to a target expression
#result = optimize(ra, target=PseudoCodeAlgebra, source=LogicalAlgebra)
result = optimize(ra, target=ProtobufAlgebra, source=LogicalAlgebra)
#result = optimize(ra, target=CCAlgebra, source=LogicalAlgebra)

print result

# generate code in the target language
print compile(result)
