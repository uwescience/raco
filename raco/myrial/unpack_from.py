"""Unpack the from clause of a bag comprehension.

This module converts a list of inputs into a single operation representing
their cross-product.  In addition, it rewrites "dotted" attribute references
to refer to the new operation's schema.  For example:

x = [FROM X, Y WHERE X.id == Y.dept_id EMIT Y.$3];

The output operation is CROSS(X, Y)
The WHERE and EMIT clauses are re-written such that dotted attribute refs
are converted into raw indexes.
"""

import raco.expression

import types

class ColumnIndexOutOfBounds(Exception):
    pass

def __calculate_offsets(from_args):
    """Calculate the first column of each relation in the rollup schema."""
    index = 0
    offsets = {}
    for _id, op in from_args.iteritems():
        offsets[_id] = index
        index += len(from_args[_id].scheme())

    return offsets

def __rewrite_expression(sexpr, from_args, base_offsets):
    """Convert all DottedAttributRefs into raw indexes."""

    def rewrite_node(sexpr):
        if not isinstance(sexpr, raco.expression.DottedAttributeRef):
            return sexpr
        else:
            op = from_args[sexpr.relation_name]
            scheme = op.scheme()

            if type(sexpr.field) == types.IntType:
                if sexpr.field >= len(scheme):
                    raise ColumnIndexOutOfBounds(str(sexpr))
                offset = sexpr.field
            else:
                offset = scheme.getPosition(sexpr.field)

            offset += base_offsets[sexpr.relation_name]
            return raco.expression.UnnamedAttributeRef(offset)

    def recursive_eval(sexpr):
        """Rewrite a node and all its descendents"""
        newexpr = rewrite_node(sexpr)
        newexpr.apply(recursive_eval)
        return newexpr

    return recursive_eval(sexpr)

def unpack(from_args, where_clause, emit_clause):
    """Proceess a list of from arguments.   Inputs:

    - from_args: A dictionary: id => raco.algebra.Operation
    - where_clause: An optional scalar expression (raco.expression)
    - emit_clause: An optional list of tuples of the form:
    (column_name, scalar_expression)

    Return values:
    - A single raco.algebra.Operation instance
    - A (possibly modified) where_clause
    - A (possibly modified) emit_clause
    """

    assert len(from_args) > 0

    def cross(x,y):
        return raco.algebra.CrossProduct(x,y)

    from_ops = from_args.values()
    op = reduce(cross, from_ops)

    offsets = __calculate_offsets(from_args)

    if where_clause:
        where_clause = __rewrite_expression(where_clause, from_args, offsets)

    if emit_clause:
        emit_clause = [(name, __rewrite_expression(sexpr, from_args, offsets))
                       for (name, sexpr) in emit_clause]

    return op, where_clause, emit_clause

