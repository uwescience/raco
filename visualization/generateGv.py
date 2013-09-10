#!/usr/bin/env PYTHONPATH=.. python

from raco import RACompiler
from raco.language import MyriaAlgebra
from raco.algebra import LogicalAlgebra

def nodes_to_id(nodes):
    return [id(x) for x in nodes]

def edges_to_id(edges):
    return [(id(x), id(y)) for (x, y) in edges]

def print_gv(nodes, edges):
    # Header
    ret = """digraph G {
  ratio = 1.333333 ;
  mincross = 2.0 ;
  rankdir = "BT" ;
  nodesep = 0.25 ;
  node [fontname="Helvetica", fontsize=10, shape=oval, style=filled, fillcolor=white ] ;

"""
    
    # Nodes
    for n in nodes:
        ret += '  "%s" [label="%s"] ;\n' % (id(n), str(n))
    ret += '\n'

    # Edges
    for (x, y) in edges:
        ret += '  "%s" -> "%s" ;\n' % (id(x), id(y))

    ret += '}'
    return ret

def main():
    # A simple join
    query = "A(x,z) :- R(x,y), S(y,z)"

    # Triangle
    # query = "A(x,z) :- R(x,y),S(y,z),T(z,x)"

    # Create a cmpiler object
    dlog = RACompiler()

    # parse the query
    dlog.fromDatalog(query)

    # Print out the graph
    for (label, root_operator) in dlog.logicalplan:
        graph = root_operator.collectGraph()
        print print_gv(graph['nodes'], graph['edges'])

if __name__ == "__main__":
    main()
