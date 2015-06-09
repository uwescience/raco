"""Functionality for handling queries with multiple relational arguments."""

from raco import algebra
from raco import expression
from raco.expression.statevar import *
from raco.myrial.exceptions import *


def rewrite_statemods(statemods, from_args, base_offsets):
    """Convert DottedRef expressions contained inside statemod variables.

    :param statemods: A list of StateVar instances
    :param from_args: A map from relation name to Operator
    :param base_offsets: A map from relation name to initial column offsets
    :return: An updated list of StateVar instances
    """
    assert all(isinstance(sm, StateVar) for sm in statemods)
    return [StateVar(name, init, rewrite_refs(update, from_args, base_offsets))
            for name, init, update in statemods]


def rewrite_refs(sexpr, from_args, base_offsets):
    """Convert all DottedRef expressions into raw indexes."""

    def rewrite_node(sexpr):
        # Push unboxing into the state variables of distributed aggregates
        if isinstance(sexpr, expression.AggregateExpression):
            if sexpr.is_decomposable():
                ds = sexpr.get_decomposable_state()
                lsms = rewrite_statemods(ds.get_local_statemods(), from_args, base_offsets)  # noqa
                rsms = rewrite_statemods(ds.get_remote_statemods(), from_args, base_offsets)  # noqa

                if lsms or rsms:
                    sexpr.set_decomposable_state(
                        expression.DecomposableAggregateState(
                            ds.get_local_emitters(), lsms,
                            ds.get_remote_emitters(), rsms,
                            ds.get_finalizer()))
                return sexpr

        if not isinstance(sexpr, expression.DottedRef):
            return sexpr
        elif sexpr.table_alias not in from_args:
            raise NoSuchRelationException(sexpr.table_alias)
        else:
            op = from_args[sexpr.table_alias]
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

            offset += base_offsets[sexpr.table_alias]
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
