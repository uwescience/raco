import collections
import types

import raco.expression

class UnboxState(object):
    def __init__(self, global_symbols, initial_pos):
        # Mapping from symbol name to Operator; read-only
        self.global_symbols = global_symbols

        # A mapping from relation name to column index
        self.local_symbols = collections.OrderedDict()

        # The next column index to be assigned
        self.pos = initial_pos

def __unbox_expression(expr, ub_state):
    def unbox_node(expr):
        if not isinstance(expr, raco.expression.Unbox):
            return expr
        else:
            # Convert the unbox operation into a simple attribute reference
            # on the forthcoming cross-product table.
            scheme = ub_state.global_symbols[expr.table].scheme()

            if not expr.table in ub_state.local_symbols:
                ub_state.local_symbols[expr.table] = ub_state.pos
                ub_state.pos += len(scheme)

            if not expr.field:
                offset = 0
            elif type(expr.field) == types.IntType:
                offset = expr.field
            else:
                # resolve name into position
                offset = scheme.getPosition(expr.field)

            return raco.expression.UnnamedAttributeRef(
                offset + ub_state.local_symbols[expr.table])

    def recursive_eval(expr):
        """Apply unbox to a node and all its descendents"""
        newexpr = unbox_node(expr)
        newexpr.apply(recursive_eval)
        return newexpr

    return recursive_eval(expr)

def unbox(op, where_clause, emit_clause, symbols):
    ub_state = UnboxState(symbols, len(op.scheme()))

    if where_clause:
        where_clause = __unbox_expression(where_clause, ub_state)

    if emit_clause:
        emit_clause = [(name, __unbox_expression(sexpr, ub_state)) for
                       (name, sexpr) in emit_clause]

    def cross(x,y):
        return raco.algebra.CrossProduct(x,y)

    # Update the op to be the cross product of all unboxed tables
    cps = [symbols[key] for key in ub_state.local_symbols.keys()]
    op = reduce(cross, cps, op)
    return op, where_clause, emit_clause
