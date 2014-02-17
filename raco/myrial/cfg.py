from raco.algebra import *

import networkx as nx

"""Control flow graph implementation.

Nodes are operations, edges are control flow.
"""

class ControlFlowGraph(object):
    def __init__(self):
        self.graph = nx.DiGraph()
        self._next_op_id = 0

    @property
    def next_op_id(self):
        return self._next_op_id

    def add_op(self, op, def_set, uses_set):
        """Add an operation to the CFG.

        :param op: A relational algebra operation (plan)
        :type op: raco.algebra.Operator
        :param def_set: Set of variables defined by the operation
        :type def_set: Set of strings
        :param uses_set: Set of variables read by the operation
        :type uses_set: Set of strings
        """

        op_id = self.next_op_id
        self._next_op_id += 1

        self.graph.add_node(op_id, op=op, defs=def_set, uses=uses_set)

        # Add a control flow edge from the prevoius statement; this assumes we
        # don't do jumps or any other non-linear control flow.
        if op_id > 0:
            self.graph.add_edge(op_id - 1, op_id)

    def add_edge(self, source, dest):
        """An an explicit edge to the control flow graph."""
        self.graph.add_edge(source, dest)

    def compute_liveness(self):
        """Run liveness analysis over the control flow graph.

        http://www.cs.colostate.edu/~mstrout/CS553/slides/lecture03.pdf

        :returns: A tuple containing live_in, live_out dictionaries.  The keys
        are variable names (strings) and the values are string sets.
        """

        num_nodes = len(self.cfg)
        live_in = collections.defaultdict(set)
        live_out = collections.defaultdict(set)

        while True:
            live_in_prev = copy.copy(line_in)
            live_out_prev = copy.copy(line_out)

            for i in range(num_nodes):
                # accessed variables are live-in
                line_in[i].update(self.cfg[i]['uses'])

                # live out variables that are not defined are live-in
                live_in[i].update(live_out_prev[i] - self.cfg[i]['defs'])

                # variables that are live-in at a successor are live-out
                for successor in nx.Digraph.successors(i):
                    live_out[i].update(live_in_prev[successor])

            if live_in == live_in_prev and live_out == live_out_prev:
                return live_in, live_out

    def dead_code_elimination(self):
        dead_set = set()

        while True:
            live_in, live_out = self.compute_liveness()
            for var, out_set in live_out.iteritems():
                defs = self.cfg[var]['defs']
                if not defs.issubset(out_set):
                    dead_set.add(var)
            self.cfg.remove_nodes_from(dead_set)
            if len(dead_set) == 0:
                break

    def get_logical_plan(self):
        """Extract a logical plan from the control flow graph.

        The logic here is simplistic:
        * Any node with in-degree == 2 is the top of a new do/while loop
        * Any node with out-degree == 2 is a while condition
        * Any other node is an ordinary operation.

        :returns: An instance of raco.algebra.Operator
        """

        op_stack = [Sequence()]
        def current_block():
            return op_stack[-1]

        for i in range(self.next_op_id):
            op = self.graph[i]['op']
            if self.graph.out_degree(i) == 2:
                # Terminate current do/while loop
                assert isinstance(current_block(), DoWhile)
                current_block().add(op)
                op_stack.pop()
                continue

            if self.graph.in_degree(i) == 2:
                # Introduce new do/while loop
                op_stack.append(DoWhile())

            current_block().add(op)

        assert len(op_stack) == 1
        return current_block()
