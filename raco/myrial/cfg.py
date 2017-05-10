from raco.algebra import *
from raco.myrial.exceptions import MyrialCompileException

import bisect
import copy
import itertools
import logging
import networkx as nx

"""Control flow graph implementation.

Nodes are operations, edges are control flow.
"""

LOG = logging.getLogger(__name__)


def find_gt(a, x):
    """Find smallest value strictly greater than x"""
    i = bisect.bisect(a, x)
    if i != len(a):
        return a[i]
    return None


def find_lt(a, x):
    """Find largest value strictly less than x"""
    i = bisect.bisect(a, x)
    if i - 1 > 0:
        return a[i - 1]
    return None


def sliding_window(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable.

    http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python  # noqa
    """

    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class ControlFlowGraph(object):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.sorted_vertices = []
        self._next_op_id = 0

    def __str__(self):
        g = self.graph
        node_strs = ['%s: uses=%s def=%s' % (str(n), repr(g.node[n]['uses']),
                                             repr(g.node[n]['def_var']))
                     for n in g]
        edge_strs = ['%s=>%s' % (str(s), str(d)) for s, d in g.edges()]
        return '; '.join(node_strs) + '\n' + '; '.join(edge_strs)

    @property
    def next_op_id(self):
        return self._next_op_id

    def add_op(self, op, _def, uses_set):
        """Add an operation to the CFG.

        :param op: A relational algebra operation (plan)
        :type op: raco.algebra.Operator
        :param _def: The variable defined by the operation or None
        :type _def: string
        :param uses_set: Set of variables read by the operation
        :type uses_set: Set of strings
        """

        op_id = self.next_op_id
        self._next_op_id += 1

        self.graph.add_node(op_id, op=op, def_var=_def, uses=uses_set)
        self.sorted_vertices.append(op_id)

        # Add a control flow edge from the previous statement; this assumes we
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

        # All variables that are accessed are live-in at a node
        live_in = {i: copy.copy(self.graph.node[i]['uses'])
                   for i in self.graph}
        live_out = {i: set() for i in self.graph}

        while True:
            # Create a copy of live_in, live_out that we will compare against
            # to detect loop termination. This is shallower than a full deep
            # copy, but we need to copy the value sets in this dictionary as
            # they are mutated in the loop below. Make the sets frozen to
            # ensure they are really immutable.
            live_in_prev = {i: frozenset(s) for i, s in live_in.items()}
            live_out_prev = {i: frozenset(s) for i, s in live_out.items()}

            for i in self.graph:
                # live out variables that are not defined are live-in
                def_var = self.graph.node[i]['def_var']
                def_set = set()
                if def_var is not None:
                    def_set.add(def_var)
                live_in[i].update(live_out_prev[i] - def_set)

                # variables that are live-in at a successor are live-out
                for successor in self.graph.successors(i):
                    live_out[i].update(live_in_prev[successor])

            if live_in == live_in_prev and live_out == live_out_prev:
                return live_in, live_out

    def __delete_node(self, node):
        """Remove a node from the control flow graph.

        Add an edge to the graph to "skip over" the target node.
        """
        assert node in self.graph

        # find a successor of the given node
        predecessors = self.graph.predecessors(node)
        if predecessors:
            successor = find_gt(self.sorted_vertices, node)
            if successor is not None:
                for p in predecessors:
                    self.graph.add_edge(p, successor)

        self.graph.remove_node(node)
        self.sorted_vertices.remove(node)

        assert node not in self.graph

    def __inline_node(self, dest_node, target_node):
        """Inline the target node into the destination node."""

        assert target_node in self.graph
        assert dest_node in self.graph

        target_op = self.graph.node[target_node]['op']
        assert isinstance(target_op, StoreTemp)
        target_inner_op = target_op.input

        dest_op = self.graph.node[dest_node]['op']

        var = self.graph.node[target_node]['def_var']
        assert var is not None

        new_op = inline_operator(dest_op, var, target_inner_op)
        self.graph.node[dest_node]['op'] = new_op

        # The merged node uses the union of the previous nodes input variables
        uses_set = self.graph.node[dest_node]['uses']
        uses_set.remove(var)
        uses_set.update(self.graph.node[target_node]['uses'])

        self.__delete_node(target_node)

    def apply_chaining(self):
        """Merge adjacent statements by chaining together plans.

        It is often desirable to chain plans instead of materializing temporary
        tables. Consider this simple example:

        X = SCAN(foo);   -- Materializing a temporary table for X would be dumb
        Y = DISTINCT(X); -- Instead, we can inline the scan into this
                         -- expression

        The merge procedure operates on the control flow graph.  We inline node
        A into node B whenever the following conditions are all true:

        - A directly precedes B; we don't consider out-of-order executions
        - A defines a variable (i.e., it assigns a temporary; not a STORE)
        - B references the variable defined by A -- def(A) in uses(B)
        - The variable defined by A is not used again
              -- def(A) not in live_out(B)
        - A and B are in the same do/while loop.

        The merge procedure is applied recursively on the CFG until convergence
        is reached.
        """

        _continue = True
        while _continue:
            live_in, live_out = self.compute_liveness()
            _continue = False

            # TODO XXX O(N^2) algorithm

            # Walk through the program backwards, and try inlining line A into
            # line B according to the above logic.
            #
            # In most cases, A is line i and B is line i+1. However, when we
            # successfully inline A into B, we want to consider inlining the
            # next A' into B, not into A.  In these cases we save the current B
            # in the inlined_into variable and then reuse it as the inline
            # candidate the next round.
            inlined_into = None
            for nodeB, nodeA in sliding_window(reversed(self.sorted_vertices)):
                if inlined_into is not None:
                    nodeB = inlined_into
                    inlined_into = None

                if self.graph.in_degree(nodeB) == 2:
                    continue  # start of do/while loop

                if isinstance(self.graph.node[nodeB]['op'], UntilConvergence):
                    continue  # start of do/until convergence

                def_var = self.graph.node[nodeA]['def_var']
                if not def_var:
                    continue

                uses = self.graph.node[nodeB]['uses']
                if def_var not in uses:
                    continue

                if def_var in live_out[nodeB]:
                    continue

                self.__inline_node(nodeB, nodeA)
                _continue = True
                inlined_into = nodeB

    def dead_code_elimination(self):
        """Dead code elimination.

        Specifically: delete CFG nodes that define a variable that is not in
        the live_out set. Recurse until convergence.
        """

        _continue = True
        while _continue:
            _continue = False
            live_in, live_out = self.compute_liveness()

            for node in self.graph:
                out_set = live_out[node]
                def_var = self.graph.node[node]['def_var']

                # Only delete nodes that 1) Define a variable (and therefore
                # aren't STORE, etc.); 2) Are not required downstream.
                if def_var and def_var not in out_set:
                    self.__delete_node(node)
                    _continue = True
                    break

    def dead_loop_elimination(self):
        """Delete entire do/while loops whose results are not consumed.

        See get_logical_plan for logic."""

        if len(self.sorted_vertices) == 0:
            return

        # A stack that contains the defined variables within each loop.
        def_set_stack = []

        def current_def_set():
            if len(def_set_stack) > 0:
                return def_set_stack[-1]
            else:
                return None

        current_loop_first_index = -1
        loops_to_delete = []  # tuples of the form [begin_index, end_index]

        live_in, live_out = self.compute_liveness()
        last_op = self.sorted_vertices[-1]

        for i in self.sorted_vertices:
            if self.graph.in_degree(i) == 2:
                # start new do/while loop
                current_loop_first_index = i
                def_set_stack.append(set())
                # Add anything defined by the current statement to the def_set
                def_var = self.graph.node[i]['def_var']
                if def_var:
                    def_set_stack[-1].add(def_var)
            elif (current_def_set() is not None and
                    (self.graph.out_degree(i) == 2 or i == last_op)):
                # end of do/while loop: check whether anything this loop
                # defines is live_in after the loop.
                def_set = def_set_stack.pop()
                next_op = find_gt(self.sorted_vertices, i)

                loop_range = (current_loop_first_index, i)

                if next_op is None:
                    # no next node?  Loop is obviously dead
                    loops_to_delete.append(loop_range)
                elif len(def_set.intersection(live_in[next_op])) == 0:
                    loops_to_delete.append(loop_range)
            elif current_def_set() is not None:
                # Add anything defined by the current statement to the def_set
                def_var = self.graph.node[i]['def_var']
                if def_var:
                    def_set_stack[-1].add(def_var)

        if not loops_to_delete:
            return

        # Delete the operations corresponding to dead loops
        for begin, end in loops_to_delete:
            for ix in range(begin, end + 1):
                self.graph.remove_node(ix)
                self.sorted_vertices.remove(ix)

            # Add a control flow edge that "skips over" the deleted loop
            if begin > 0 and end < last_op:
                prev = find_lt(self.sorted_vertices, begin)
                _next = find_gt(self.sorted_vertices, end)

                assert prev in self.graph
                assert _next in self.graph
                assert prev < begin
                assert _next > end
                self.graph.add_edge(prev, _next)

        self.dead_loop_elimination()

    def get_logical_plan(self, dead_code_elimination=True,
                         apply_chaining=True):
        """Extract a logical plan from the control flow graph.

        The logic here is simplistic:
        * Any node with in-degree == 2 is the top of a new do/while loop
        * Any node with out-degree == 2 is a while condition
        * Any other node is an ordinary operation.

        :returns: An instance of raco.algebra.Operator
        """

        if dead_code_elimination:
            self.dead_loop_elimination()
            self.dead_code_elimination()

        if apply_chaining:
            self.apply_chaining()

        if not self.graph:
            raise MyrialCompileException("Optimized program is empty")

        op_stack = [Sequence()]

        def current_block():
            return op_stack[-1]

        for i in sorted(self.graph):
            op = self.graph.node[i]['op']

            if self.graph.out_degree(i) == 2:
                LOG.info("Terminating while loop (%d): %s", i, op)
                # Terminate current do/while loop
                assert isinstance(current_block(), DoWhile)
                current_block().add(op)

                do_while_op = op_stack.pop()
                current_block().add(do_while_op)
                continue

            if self.graph.in_degree(i) == 2:
                LOG.info("Introducing while loop (%d)", i)
                # Introduce new do/while loop
                op_stack.append(DoWhile())

            LOG.info("Adding operation to sequence (%d) %s", i, op)
            current_block().add(op)

        assert len(op_stack) == 1
        return current_block()
