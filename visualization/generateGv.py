#!/usr/bin/env PYTHONPATH=.. python

from raco import RACompiler
from raco.language import MyriaAlgebra
from raco.algebra import LogicalAlgebra
from raco.viz import graph_to_dot, operator_to_dot, plan_to_dot

def main():
    # A simple join
    query = "A(x,z) :- R(x,y), S(y,z)"

    # Triangle
    # query = "A(x,z) :- R(x,y),S(y,z),T(z,x)"

    # Two independent rules
    query = """A(x,z) :- R(x,y), S(y,z); B(x,z) :- S(z,x)."""

    # Create a cmpiler object
    dlog = RACompiler()

    # parse the query
    dlog.fromDatalog(query)

    # Print out the graph
    for (label, root_operator) in dlog.logicalplan:
        graph = root_operator.collectGraph()
        v1 = graph_to_dot(graph)
        v2 = operator_to_dot(root_operator)
        assert v1 == v2
        print "Dot for individual IDB %s: " % label
        print v1
        print

    v3 = plan_to_dot(dlog.logicalplan)
    print "Dot for combined IDBs:"
    print v3
    print
    if len(dlog.logicalplan) == 1:
        assert v1 == v3

if __name__ == "__main__":
    main()
