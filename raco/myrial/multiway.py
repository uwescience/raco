"""Functionality for handling queries with multiple relational arguments."""

from raco import algebra
from raco import expression
from raco.myrial.exceptions import ColumnIndexOutOfBounds


def rewrite_refs(sexpr, from_args, base_offsets):
    """Convert all Unbox expressions into raw indexes."""

    def rewrite_node(sexpr):
        if not isinstance(sexpr, expression.Unbox):
            return sexpr
        else:
            op = from_args[sexpr.relational_expression]
            scheme = op.scheme()

            debug_info = None
            if not sexpr.field:
                offset = 0
            elif isinstance(sexpr.field, int):
                if sexpr.field >= len(scheme):
                    raise ColumnIndexOutOfBounds(str(sexpr))
                offset = sexpr.field
            else:
                assert isinstance(sexpr.field, basestring)
                offset = scheme.getPosition(sexpr.field)
                debug_info = sexpr.field

            offset += base_offsets[sexpr.relational_expression]
            return expression.UnnamedAttributeRef(offset, debug_info)

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
        return algebra.CrossProduct(x, y)

    from_ops = from_args.values()
    op = reduce(cross, from_ops)

    return (op, __calculate_offsets(from_args))
