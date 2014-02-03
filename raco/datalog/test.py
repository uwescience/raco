import raco.datalog as dlog

programtest = """
A(X,Y) :- R(W,X,Y), R(Y,Z), Z=3.3, Z>2, X=Z, W<Z
B(X,Y) :- R(X,Y), R(Y,Z), 3=4, Y=Z
B(X,Y) :- A(X,Y), R(Y,Z), C(Y,Y), X>2, X=Z
C(X,Y) :- A(X,Y,Y,Z),Z=Z,X=Z
"""

triangle = """
A(X) :- R(Z,X), S(X,Y), T(Y,Z)
"""

debugquery = """
S(X,Y) :- F(X,Z),G(Z,Y)
A(X,Z) :- R(W,X,Y), S(Z,Y), W=3
"""

ruletest = """
A(X) :- A(Y,X), B(Y,Z), C(W,Z,P), D(W,N), E(P,M), P=4, Z<9
"""

ruletest2 = """
A(X) :- A(Y,X), B(W,Z),W=3
"""

# Join(5, 0, Join(3, 0, Join(1, 0, "R", "A"), "B"), "R")
# RA = Join(1, 0, "R", "A")
# RAB = Join(3, 0, RA, "B")
# RABR = Select(Eq(0,5), RAB)

bodytest = """
A(X,Y), B(Y,Z)
"""

termtest = """
AWord(G,A,B,"foo", 5, 3.2)
"""


def main(query):
    print "Input:"
    print query
    parsedprogram = dlog.program.parseString(query)[0]
    print "Parsed:"
    print parsedprogram
    print ""
    print "As Relational Algebra:"
    for rule in parsedprogram.rules:
        print "Rule:", rule
    print rule.toRA(parsedprogram)

if __name__ == '__main__':
    main(debugquery)
