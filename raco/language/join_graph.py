import networkx as nx

from raco.datastructure.ordered_set import OrderedSet


class JoinGraph(object):
    """Represents one or more joins.

    Nodes represent relations; edges represent equijoin conditions.
    """
    def __init__(self, node_data=[]):
        """Initialize a join graph."""
        assert len(node_data) >= 1
        self.graph = nx.MultiGraph()
        for i, data in enumerate(node_data):
            self.graph.add_node(i, data=data)

    def __len__(self):
        return len(self.graph)

    def add_edge(self, src_node, dst_node, data):
        """Add an edge representing an equijoin to the join graph."""
        assert 0 <= src_node < len(self.graph)
        assert 0 <= dst_node < len(self.graph)

        self.graph.add_edge(src_node, dst_node, data=data)

    def choose_left_deep_join_order(self):
        """Chose a left-deep join order.

        Currently, the only goal is to avoid cross-products.
        """

        joined_nodes = OrderedSet()
        graph = self.graph.copy()
        all_nodes = set(self.graph.nodes())

        while len(joined_nodes) < len(graph):
            # Add an arbitrary node to the join set
            for n in all_nodes - set(joined_nodes):
                joined_nodes.add(n)
                break

            # Expand the join set to include all reachable nodes.
            while True:
                new_nodes = set()
                old_len = len(joined_nodes)

                for n1 in joined_nodes:
                    new_nodes |= set(self.graph.neighbors_iter(n1))
                joined_nodes |= new_nodes

                if len(joined_nodes) == old_len:
                    break

        return list(joined_nodes)

    def get_edges(self, node1, node2):
        """Return a set of edge data between the given nodes."""
        if node2 in self.graph.neighbors(node1):
            return {d['data'] for d in self.graph[node1][node2].itervalues()}
        else:
            return set()
