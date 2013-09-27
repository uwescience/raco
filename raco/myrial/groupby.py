
"""
Implemention of groupby/aggregation in Myrial.

The inputs are a child Operator and a list of column mappings, each defined
by tuples of the form: (column_name, scalar_expression).  The outputs are
(possibly) revised operators and list of column mappings.

The basic algorithm:

1) Scan the list of columns looking for any aggregte expressions.  If none
are found, then we return the inputs unmodified.

2) Next, we scan each column, switching on whether the scalar expression
contains any aggregate expression:

2A) Columns without aggregate expressions are "groupby" terms.  We add such
terms to a list of groupby terms.

2B) Columns with aggregate expressions are "aggregation" terms.  We record all
aggregate expressions in a list (actually, an OrderedDict).  And, we apply
a "hoisting" procedure whereby each aggregate expression is replaced by
a reference to a raw column reference.  As an example:

range = max(salary) - min(salary)
==>
range = $4 - $5

3) We create a GroupBy relational algebra operation with the grouping terms
and aggregate terms calculated in steps 2A and 2B.

4) We return an updated set of column mappings; the Myrial interpreter
uses these mappings to form an Apply operator that stitches up the column
names and values as expected by the caller.
"""

import collections
import raco.expression

class NestedAggregateException(Exception):
    '''Nested aggregate functions are not allowed'''
    pass

def __aggregate_count(sexpr):
    def count(sexpr):
        if isinstance(sexpr, raco.expression.AggregateExpression):
            return 1
        return 0
    return sum(sexpr.postorder(count))

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
            return sexpr
        elif sexpr in gb_state.aggregates:
            return gb_state.aggregates[sexpr];
        else:
            # Check for nested aggregate expressions
            if __aggregate_count(sexpr) != 1:
                raise NestedAggregateException(str(sexpr))

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

    # TODO: Need to resolve this with the new group by 
    #op = raco.algebra.GroupBy(grouping_terms, state.aggregates.keys(), op)
    columnlist = grouping_terms + state.aggregates.keys()
    op = raco.algebra.GroupBy(columnlist, op)
    return op, output_mappings
