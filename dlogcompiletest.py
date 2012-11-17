from raco.datalog.grammar import parse
from raco.scheme import Scheme
from raco.catalog import ASCIIFile
from raco.language import PythonAlgebra, PseudoCodeAlgebra, CCAlgebra#, ProtobufAlgebra
from raco.algebra import LogicalAlgebra
from raco.compile import compile, optimize, common_subexpression_elimination, showids

# declare the schema for each relation
sch = Scheme([("subject", int), ("predicate", int), ("object", int)])

# Create a relation object.  We can add formats here as needed.
R = ASCIIFile("R", sch)

# Now write the Datalog expression
#query = """A(x,y,z) :- R(x,p1,y,t1),R(y,p2,z,t2)"""
#query = """A(x,y,z) :- R(x,p1,y,t1),R(y,p2,z,t2),R(z,p3,w,t3),t3 = 1928433933)"""
#query = """A(s1,o4) :- R(s1,'blue',o1),R(o1,'red',o2),R(o2,'green',o3),R(o3,'blue',o4)"""
#query = """A(x,z) :- R(x,y),S(y,z,a,b,c,d),T(c,x)"""
#query = """A(x,y) :- S(x), R(y,z), T(z,a)"""
#query = """A(x) :- R(x),R(x)"""
#query = """A(s,p,o,t) :- T(s,p,o,t), o=65240523"""
#query = """A(x) :- R(x,y),S(a,y,z,4),T(c,d,z,w),w<3"""
#query = """A(x) :- R(x,y),S(a,y,z),T(c,d,z)"""
#query = """A(x) :- R(x,y),S(y,z,4),z<3"""
#query = """A(x) :- S(y,z,4)"""
#query = """A(x,z) :- R(x,y,u),R(y,z,w),w=3,u=4"""
#query = """A(x,z) :- R(x,y,4),S(y,z,5),T(z,a,6),U(a,b,7),V(b,c,8)"""
#query = """A(A,B,C) :- R(A, x, B), R(A, y, C), R(C, z, B)"""
#query = """A(A,B,C) :- R(A, x, B), R(A, y, C)"""
query= """
Q2(p, auth, bookt, year, xref, ee, title, pages, url, abs) :- 

IsProceedings(p), 
Author(p, auth),
Booktitle(p, bookt), 
Year(p, year), 
Crossref(p, xref), 
Ee(p, ee), 
Title(p, title), 
Page(p, pages), 
Url(p, url), 
Abstract(p, abs)
"""
query = """
Q2(p, auth, url,ee) :- 

IsProceedings(p), 
Author(p, auth),
Url(p, url),
Ee(p, ee), 
"""
query = """
A(x,z) :- R(x,y,z)
B(w) :- A(3,w)
C(x) :- R(y,x,z), S(z,g)
"""

def comment(s):
  print "/*\n%s\n*/" % s

comment(query)

# Now parse it
parsedprogram = parse(query)

exprs = parsedprogram.toRA()
# generate an RA expression
#rules = parsedprogram.rules
#ra = onlyrule.toRA(parsedprogram)

print exprs
ra = exprs[0]

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
compile(physicalplan)
