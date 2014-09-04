
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
from raco.myrial.exceptions import *


class NonGroupedAccessException(Exception):
    """Attempting to access a non-grouping term in an aggregate expression"""
    pass


class AggregateState(object):
    def __init__(self, initial_aggregate_pos):
        # Mapping from an aggregate scalar expression (e.g., MAX(salary)) to
        # a non-aggregate scalar expression (a raw column index).
        self.aggregates = collections.OrderedDict()

        # Next index to be assigned for aggregate expressions
        self.aggregate_pos = initial_aggregate_pos


def __hoist_aggregates(sexpr, agg_state, group_mappings, input_scheme):
    def hoist_node(sexpr):
        if isinstance(sexpr, raco.expression.AttributeRef):
            # Translate the attribute ref to the schema that will exist
            # after the GroupBy
            input_pos = sexpr.get_position(input_scheme)
            if input_pos not in group_mappings:
                raise NonGroupedAccessException(str(sexpr))
            output_pos = group_mappings[input_pos]
            return raco.expression.UnnamedAttributeRef(output_pos)

        if not isinstance(sexpr, raco.expression.AggregateExpression):
            return sexpr

        if sexpr in agg_state.aggregates:
            return agg_state.aggregates[sexpr]
        else:
            out = raco.expression.UnnamedAttributeRef(agg_state.aggregate_pos)
            agg_state.aggregates[sexpr] = out
            agg_state.aggregate_pos += 1
            return out

    def recursive_eval(sexpr):
        """Apply hoisting to a scalar expression and all its descendents"""
        newexpr = hoist_node(sexpr)
        newexpr.apply(recursive_eval)
        return newexpr

    return recursive_eval(sexpr)


def groupby(op, emit_clause, extra_grouping_columns, statemods=None):
    """Process groupby/aggregation expressions."""

    assert emit_clause

    # A mapping from input position (before the GroupBy) to output position
    # (after the GroupBy) for grouping terms.  This allows aggregate terms
    # to refer to grouping fields.
    group_mappings = {}

    scheme = op.scheme()
    num_group_terms = 0

    for name, sexpr in emit_clause:
        if not raco.expression.expression_contains_aggregate(sexpr):
            if isinstance(sexpr, raco.expression.AttributeRef):
                group_mappings[sexpr.get_position(scheme)] = num_group_terms
            num_group_terms += 1

    # The user must have specified an aggregate expression to trigger
    # a groupby invocation.
    assert num_group_terms != len(emit_clause)

    # Add extra grouping columns; we group by these terms, but the output
    # is not preserved in the final apply invocation.  These are columns
    # that were referenced in unbox expressions.
    for col in extra_grouping_columns:
        group_mappings[col] = num_group_terms
        num_group_terms += 1

    # State about scalar expressions with aggregates
    agg_state = AggregateState(num_group_terms)

    # mappings from column name to scalar expressions; these mappings are
    # applied after the GroupBy operator to stitch up column names and values.
    output_mappings = []

    # A subset of the scalar expressions in the emit clause that do
    # not contain aggregate expressions; these become the grouping terms
    # to the GroupBy operator.
    group_terms = []

    for name, sexpr in emit_clause:
        if raco.expression.expression_contains_aggregate(sexpr):
            output_mappings.append(
                (name, __hoist_aggregates(sexpr, agg_state, group_mappings,
                                          scheme)))
        else:
            output_mappings.append((
                name, raco.expression.UnnamedAttributeRef(len(group_terms))))
            group_terms.append(sexpr)

    # Add extra grouping columns; note that these are not present in the
    # output mappings.
    group_terms.extend([raco.expression.UnnamedAttributeRef(c)
                        for c in extra_grouping_columns])

    agg_terms = agg_state.aggregates.keys()
    op1 = raco.algebra.GroupBy(group_terms, agg_terms, op, statemods)
    return raco.algebra.Apply(emitters=output_mappings, input=op1)
