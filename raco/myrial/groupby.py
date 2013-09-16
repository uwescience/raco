import collections

import raco.expression

class GroupbyState(object):
    def __init__(self, initial_aggregate_pos):
        # Mapping from an aggregate scalar expression (e.g., MAX(salary)) to
        # a non-aggregate scalar expression (a raw column index).
        self.aggregates = collections.OrderedDict()

        # Next index to be assigned for aggregate expressions
        self.aggregate_pos = initial_aggregate_pos

def __hoist_aggregates(sexpr, gb_state):
    def hoist_node(sexpr):
        if not isinstance(sexpr, raco.expression.AggregateExpression):
            return expr
        elif sexpr in gb_state.aggregates:
            return gb_state.aggregates[sexpr];
        else:
            out = raco.expression.UnnamedAttributeRef(gb_state.aggregate_pos)
            gb_state.aggregates[sexpr] = out
            gb_state.aggregate_pos += 1
            return out

    def recursive_eval(sexpr):
        """Apply hoisting to a scalar expression and all its descendents"""
        newexpr = hoist_node(sexpr)
        newexpr.apply(recursive_eval)
        return newexpr

    return recursive_eval(sexpr)

def sexpr_contains_aggregate(sexpr):
    """Return 1 if a scalar expression contains 1 or more aggregates"""
    def is_aggregate(sexpr):
        return isinstance(sexpr, raco.expression.AggregateExpression)

    if any(sexpr.postorder(is_aggregate)):
        return 1
    else:
        return 0

def groupby(op, emit_clause):
    """Process any groupby/aggregation expressions."""

    if not emit_clause:
        return op, emit_clause

    # Perform a simple count of output columns with aggregate expressions
    num_agg_columns = sum([sexpr_contains_aggregate(sexpr) for _, sexpr in
                           emit_clause])

    if num_agg_columns == 0:
        return op, emit_clause # No aggregates: not a groupby query

    state = GroupbyState(len(emit_clause) - num_agg_columns)
    output_mappings = [] # mappings from column name to sexpr
    grouping_terms = [] # list of sexpr without aggregate functions

    for name, sexpr in emit_clause:
        if sexpr_contains_aggregate(sexpr):
            output_mappings.append((name,  __hoist_aggregates(sexpr, state)))
        else:
            out = raco.expression.UnnamedAttributeRef(len(grouping_terms))
            output_mappings.append((name, out))
            grouping_terms.append(sexpr)

    op = raco.algebra.GroupBy(grouping_terms, gb_state.aggregates.keys(), op)
    return op, output_mappings
