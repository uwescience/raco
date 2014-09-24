"""
Utility functions for use in Raco expressions
"""

from .expression import (Expression, BinaryOperator, AttributeRef,
                         NamedAttributeRef, UnnamedAttributeRef,
                         NamedStateAttributeRef)
from .aggregate import BuiltinAggregateExpression, AggregateExpression

import copy
import inspect


class NestedAggregateException(Exception):
    """Nested aggregate functions are not allowed"""
    def __init__(self, lineno):
        self.lineno = lineno

    def __str__(self):
        return "Nested aggregate expression on line %d" % self.lineno


def ensure_unnamed(exprs, op):
    """Returns the expressions in the supplied exprs, ensuring that all
    attribute references are in the unnamed perspective. The expressions are
    resolved against the scheme of the supplied operator, but only if
    necessary.

    :param exprs: an expression or sequence of expressions.
    :param op: the operator against whose scheme non-unnamed references will be
               resolved. Scheme is only computed if needed.
    """
    if isinstance(exprs, Expression):
        # exprs is just a single expression
        if only_unnamed_refs(exprs):
            return exprs
        return to_unnamed_recursive(exprs, op.scheme())

    else:
        # exprs is a sequence of expressions.
        if all(only_unnamed_refs(e) for e in exprs):
            return exprs
        op_scheme = op.scheme()
        return [to_unnamed_recursive(e, op_scheme) for e in exprs]


def toUnnamed(ref, scheme):
    """Convert a reference to the unnamed perspective"""
    if issubclass(ref.__class__, UnnamedAttributeRef):
        return ref
    elif issubclass(ref.__class__, NamedAttributeRef):
        return UnnamedAttributeRef(scheme.getPosition(ref.name))
    else:
        raise TypeError("Unknown value reference %s.  Expected a position reference or an attribute reference." % ref)  # noqa


def toNamed(ref, scheme):
    """Convert a reference to the named perspective"""
    if issubclass(ref.__class__, UnnamedAttributeRef):
        attrname = scheme[ref.position][0]
        return NamedAttributeRef(attrname)
    elif issubclass(ref.__class__, NamedAttributeRef):
        return ref
    else:
        raise TypeError("Unknown value reference %s.  Expected a position reference or an attribute reference.")  # noqa


def to_unnamed_recursive(sexpr, scheme):
    """Convert all named column references to unnamed column references."""
    def convert(n):
        if isinstance(n, NamedAttributeRef):
            n = toUnnamed(n, scheme)
        n.apply(convert)
        return n
    return convert(sexpr)


def only_unnamed_refs(sexpr):
    """Returns True if all AttributeRefs in the expression are unnamed."""
    return all(isinstance(e, UnnamedAttributeRef)
               for e in sexpr.walk()
               if isinstance(e, AttributeRef))


def all_classes():
    """Return a list of all classes in the module"""
    import raco.expression as expr
    return [obj for _, obj in inspect.getmembers(expr, inspect.isclass)]


def aggregate_functions():
    """Return all the classes that can be used to construct an aggregate expression"""  # noqa
    allclasses = all_classes()
    opclasses = [opclass for opclass in allclasses
                 if issubclass(opclass, BuiltinAggregateExpression)
                 and not inspect.isabstract(opclass)]

    return opclasses


def binary_ops():
    """Return a list of all classes used to construct arithmetic, like PLUS, DIVIDE, etc."""  # noqa
    allclasses = all_classes()
    opclasses = [opclass for opclass in allclasses
                 if issubclass(opclass, BinaryOperator)
                 and not inspect.isabstract(opclass)]
    return opclasses


def udf_undefined_vars(expr, vars):
    """Return a list of undefined variables in a UDF.

    :param expr: An expression corresponding to a UDF.  Variable references are
    identified by instances of NamedAttributeRef.

    :param vars: A list of variables in the argument list to the function.
    :type vars: list of strings
    """
    return [ex.name for ex in expr.walk()
            if isinstance(ex, NamedAttributeRef) and ex.name not in vars]


def resolve_function(func_expr, arg_dict):
    """Bind variables to arguments in a function invocation.

    :param func_expr: An expression corresponding to function
    :type func_expr: Expression
    :param arg_dict: The arguments to the function
    :type arg_dict: A dictionary mapping string to Expression
    :returns: An expression with no variables
    """

    def convert(n):
        if isinstance(n, NamedAttributeRef):
            n = copy.deepcopy(arg_dict[n.name])
        else:
            n.apply(convert)
        return n

    return convert(copy.deepcopy(func_expr))


def resolve_state_vars(expr, state_vars, mangled_names):
    """Convert references to state variables into NamedStateAttributeRef
    references.

    :param expr: An expression instances
    :type expr: Expression
    :param state_vars: Variables that represent state variables
    :type state_vars: List of strings
    :param mangled_names: A mapping from name to its mangled version
    :type mangled_names: Dictionary from string to string
    :return: An instance of Expression
    """

    def convert(n):
        if isinstance(n, NamedAttributeRef) and n.name in state_vars:
            n = NamedStateAttributeRef(mangled_names[n.name])
        else:
            n.apply(convert)
        return n

    return convert(copy.deepcopy(expr))


def accessed_columns(expr):
    """Return a set of column indexes accessed by an expression.

    Assumes that named attribute references have been converted to integer
    positions.
    """
    for ex in expr.walk():
        assert not isinstance(ex, NamedAttributeRef)

    return set([ex.position for ex in expr.walk()
                if isinstance(ex, UnnamedAttributeRef)])


def rebase_expr(expr, offset):
    """Subtract the given offset from each column access.

    Assumes that named attribute references have been converted to integer
    positions.
    """
    assert offset > 0

    for ex in expr.walk():
        assert not isinstance(ex, NamedAttributeRef)
        if isinstance(ex, UnnamedAttributeRef):
            ex.position -= offset


def reindex_expr(expr, index_map):
    """Changes references to key columns to references to value columns in
    index_map.

    Assumes that named attribute references have been converted to integer
    positions.
    """
    for ex in expr.walk():
        assert (not isinstance(ex, AttributeRef)
                or isinstance(ex, UnnamedAttributeRef))
        if isinstance(ex, UnnamedAttributeRef) and ex.position in index_map:
            ex.position = index_map[ex.position]


def expression_contains_aggregate(ex):
    """Return True if the expression contains an aggregate."""
    return any(isinstance(sx, AggregateExpression) for sx in ex.walk())


def check_no_aggregate(ex, lineno):
    """Raise an exception if the provided expression contains an aggregate."""
    if expression_contains_aggregate(ex):
        raise NestedAggregateException(lineno)


def check_no_nested_aggregate(ex, lineno):
    """Raise an exception if the expression contains a nested aggregate."""

    def descend(sx, in_aggregate):
        is_aggregate = isinstance(sx, AggregateExpression)
        if is_aggregate and in_aggregate:
            raise NestedAggregateException(lineno)
        for child in sx.get_children():
            descend(child, is_aggregate or in_aggregate)

    descend(ex, False)
