def graph_to_dot(graph):
    """Graph is expected to be a dict of the form { 'nodes' : list(), 'edges' :
    list() }. This function returns a string that will be input to dot."""

    # Template, including setup and formatting:
    template = """digraph G {
      size = "3.0, 4.0" ;
      ratio = "fill" ;
      mincross = 2.0 ;
      rankdir = "BT" ;
      nodesep = 0.25 ;
      node [fontname="Helvetica", fontsize=10, shape=oval, style=filled, fillcolor=white ] ;

      // The nodes
      %s

      // The edges
      %s
}"""

    # Nodes:
    nodes = ['"%s" [label="%s"] ;' % (id(n), n.shortStr()) for n in graph['nodes']]
    node_str = '\n      '.join(nodes)

    # Edges:
    edges = ['"%s" -> "%s" ;' % (id(x), id(y)) for (x, y) in graph['edges']]
    edge_str = '\n      '.join(edges)

    return template % (node_str, edge_str)

def operator_to_dot(operator, graph=None):
    """Operator is expected to be an object of class raco.algebra.Operator"""
    graph = operator.collectGraph(graph)
    return graph_to_dot(graph)

def plan_to_dot(label_op_list):
    """label_op_list is expected to be a list of [('Label', Operator)] pairs
    where Operator is of type raco.algebra.Operator"""

    graph = None
    for (label, root_operator) in label_op_list:
        graph = root_operator.collectGraph(graph)
    return graph_to_dot(graph)
