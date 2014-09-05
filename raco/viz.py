from raco import algebra


def graph_to_dot(graph, **kwargs):
    """Graph is expected to be a dict of the form { 'nodes' : list(), 'edges' :
    list() }. This function returns a string that will be input to dot."""

    title = kwargs.get('title', '')

    # Template, including setup and formatting:
    template = """digraph G {
      ratio = "fill" ;
      size = "4.0, 4.0" ;
      page = "4, 4" ;
      margin = 0 ;
      mincross = 2.0 ;
      rankdir = "BT" ;
      nodesep = 0.25 ;
      ranksep = 0.25 ;
      node [fontname="Helvetica", fontsize=10,
            shape=oval, style=filled, fillcolor=white ] ;

      // The nodes
      %s

      // The edges
      %s

      // The title
      labelloc="t";
      label="%s";
}"""

    # Nodes:
    nodes = ['"%s" [label="%s"] ;' % (id(n), n.shortStr().replace(r'"', r'\"'))
             for n in graph['nodes']]
    node_str = '\n      '.join(nodes)

    # Edges:
    edges = ['"%s" -> "%s" ;' % (id(x), id(y)) for (x, y) in graph['edges']]
    edge_str = '\n      '.join(edges)

    return template % (node_str, edge_str, title)


def operator_to_dot(operator, graph=None, **kwargs):
    """Operator is expected to be an object of class raco.algebra.Operator"""
    graph = operator.collectGraph(graph)
    return graph_to_dot(graph, **kwargs)


def get_dot(obj):
    if isinstance(obj, dict) and 'nodes' in dict and 'edges' in dict:
        return graph_to_dot(obj)
    elif isinstance(obj, algebra.Operator):
        return operator_to_dot(obj)
    raise NotImplementedError('Unable to get dot from object type %s' % type(obj))  # noqa
