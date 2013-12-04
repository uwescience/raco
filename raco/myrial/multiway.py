"""Functionality for handling queries with multiple relational arguments."""

import raco.expression
from raco.myrial.exceptions import ColumnIndexOutOfBounds

import types

def rewrite_refs(sexpr, from_args, base_offsets):
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

def __calculate_offsets(from_args):
    """Calculate the first column of each relation in the rollup schema."""
    index = 0
    offsets = {}
    for _id in from_args.iterkeys():
        offsets[_id] = index
        index += len(from_args[_id].scheme())

    return offsets

def merge(from_args):
    """Merge a sequence of operations into a cross-product tree.

    from_args: A dictionary mapping a unique string id to a
    raco.algebra.Operation instance.

    Returns: a single raco.algebra.Operation instance and an opaque
    data structure suitable for passing to the rewrite_refs function.
    """

    assert len(from_args) > 0

    def cross(x, y):
        return raco.algebra.CrossProduct(x, y)

    from_ops = from_args.values()
    op = reduce(cross, from_ops)

    return (op, __calculate_offsets(from_args))
