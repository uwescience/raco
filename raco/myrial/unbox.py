import collections

import raco.expression

class UnboxState(object):
    def __init__(self, initial_pos):
        # The next column index to be assigned
        self.pos = initial_pos

        # A mapping from relation name to column index
        self.symbols = collections.OrderedDict()

def __unbox_expression(expr, ub_state):
    def unbox_node(expr):
        if not isinstance(expr, raco.expression.Unbox):
            return expr
        else:
            # Convert the unbox operation into a simple attribute reference
            # on the forthcoming cross-product table.
            if not expr.table in ub_state.symbols:
                ub_state.symbols[expr.table] = ub_state.pos
                ub_state.pos += 1

            return raco.expression.UnnamedAttributeRef(
                ub_state.symbols[expr.table])

    def recursive_eval(expr):
        """Apply unbox to a node and all its descendents"""
        newexpr = unbox_node(expr)
        newexpr.apply(recursive_eval)
        return newexpr

    return recursive_eval(expr)

def unbox(op, where_clause, initial_pos, symbols):
    ub_state = UnboxState(initial_pos)

    if where_clause:
        where_clause = __unbox_expression(where_clause, ub_state)

    def cross(x,y):
        return raco.algebra.CrossProduct(x,y)

    # Update the op to be the cross product of all unboxed tables
    cps = [symbols[key] for key in ub_state.symbols.keys()]
    op = reduce(cross, cps, op)
    return op, where_clause
