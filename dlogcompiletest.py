from raco import RACompiler
from raco.scheme import Scheme
from raco.catalog import ASCIIFile
from raco.language import PythonAlgebra, PseudoCodeAlgebra, CCAlgebra#, ProtobufAlgebra
from raco.language import MyriaAlgebra
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
query1 = """
A(x,z) :- R(x,y,z)
B(w) :- A(3,w)
C(x) :- R(y,x,z), S(z,g)
"""

query = """
A(x,z) :- R(x,z)
B(x,z) :- A(x,y), R(y,z)
"""

query2 = """
A(z) :- R(x,y,z)
B(z) :- S(z,w), R(x,y,z)
"""

query2 = """
A(x,y) :- R(x,y),S(y,x)
"""

query1 = """
A@h(x,y)(x,z) :- R(x,y,z)
"""

query1 = """
# Start by randomly assigning a point to a cluster (local computation)
   PC(pid, c) :- P(pid, x), c = rand{1,.., k}

# Compute the cluster centers. First, compute the server-local cluster centers
   CL(cid, avg(x) ) :- C(pid, cid), P(pid, x)

# Broadcast the local cid's (the @* notation means that the data is sent to all servers)
   CL(@*, cid, a) :- CL(cid, a)

# Average the local averages: this is equal to the total average
   C(cid, avg(a)) :- CL(cid, a)

# We have now computed the cluster centers. Next, we assign each point to the closest cluster center. This is an *update* step. The computation is local, since C is in every server.
   PC(pid, argmin_{cid}(d(x, a)))@next :- P(pid, x), C(cid, a)

# Finally, we need a stopping condition. Here, we say that the iterations stop as soon as each cluster center has moved at most E distance in consecutive iterations. We do not have a fixed notation for this, but @prev denotes the version of the relations with the previous timestamp.
   NoStop :- C(cid, a), C(cid, b)@prev, d(a,b) > E
"""
query1 = """
# Start by randomly assigning a point to a cluster (local computation)
   PC(pid, c) :- P(pid, x), RandK(c)

# Compute the cluster centers. First, compute the server-local cluster centers
   CL(cid, avg(x) ) :- C(pid, cid), P(pid, x)

"""

query = """
A@*(x) :- R(x,y,z)
"""

query = """
smallTableJoin(x,z) :- smallTable(x,y),smallTable(y,z)
"""

def comment(s):
  print "/*\n%s\n*/" % str(s)

dlog = RACompiler()

dlog.fromDatalog(query)
print dlog.logicalplan

dlog.optimize(target=MyriaAlgebra, eliminate_common_subexpressions=False)

code = dlog.compile()
print code

# generate code in the target language
#print compile(physicalplan)
#compile(physicalplan)
print compile(physicalplan)
