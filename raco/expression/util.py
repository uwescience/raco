"""
Utility functions for use in Raco expressions
"""

from .expression import BinaryOperator, NamedAttributeRef, UnnamedAttributeRef, NamedStateAttributeRef
from .aggregate import AggregateExpression

import copy
import inspect

def toUnnamed(ref, scheme):
    """Convert a reference to the unnamed perspective"""
    if issubclass(ref.__class__, UnnamedAttributeRef):
        return ref
    elif issubclass(ref.__class__, NamedAttributeRef):
        return UnnamedAttributeRef(scheme.getPosition(ref.name))
    else:
        raise TypeError("Unknown value reference %s.  Expected a position reference or an attribute reference." % ref)

def toNamed(ref, scheme):
    """Convert a reference to the named perspective"""
    if issubclass(ref.__class__, UnnamedAttributeRef):
        attrname = scheme[ref.position][0]
        return NamedAttributeRef(attrname)
    elif issubclass(ref.__class__, NamedAttributeRef):
        return ref
    else:
        raise TypeError("Unknown value reference %s.  Expected a position reference or an attribute reference.")

def to_unnamed_recursive(sexpr, scheme):
    """Convert all named column references to unnamed column references."""
    def convert(n):
        if isinstance(n, NamedAttributeRef):
            n = toUnnamed(n, scheme)
        n.apply(convert)
        return n
    return convert(sexpr)

def all_classes():
    """Return a list of all classes in the module"""
    import raco.expression as expr
    return [obj for _, obj in inspect.getmembers(expr, inspect.isclass)]

def aggregate_functions():
    """Return all the classes that can be used to construct an aggregate expression"""
    allclasses = all_classes()
    opclasses = [opclass for opclass in allclasses
                     if issubclass(opclass, AggregateExpression)
                     and not inspect.isabstract(opclass)]

    return opclasses

def binary_ops():
    """Return a list of all classes used to construct arithmetic, like PLUS, DIVIDE, etc."""
    allclasses = all_classes()
    opclasses = [opclass for opclass in allclasses
                     if issubclass(opclass, BinaryOperator)
                     and not inspect.isabstract(opclass)]
    return opclasses

def isaggregate(expr):
    return any(expr.postorder(lambda x: isinstance(x, AggregateExpression)))

def udf_undefined_vars(expr, vars):
    """Return a list of undefined variables in a UDF.

    :param expr: An expression corresponding to a UDF.  Variable references are identified
    by instances of NamedAttributeRef.

    :param vars: A list of variables in the argument list to the function.
    :type vars: list of strings
    """
    return [ex.name for ex in expr.walk()
            if isinstance(ex, NamedAttributeRef) and ex.name not in vars]

def resolve_udf(udf_expr, arg_dict):
    """Bind variables to arguments in a UDF expression.

    :param udf_expr: An expression corresponding to UDF
    :type upf_expr: Expresison
    :param arg_dict: The arguments to the UDF
    :type arg_dict: A dictionary mapping string to Expression
    :returns: An expression with no variables
    """

    def convert(n):
        if isinstance(n, NamedAttributeRef):
            n = arg_dict[n.name]
        n.apply(convert)
        return n

    return convert(copy.copy(udf_expr))

def resolve_state_vars(expr, state_vars, mangled_names):
    """Convert references to state variables into NamedStateAttributeRef references.

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
        n.apply(convert)
        return n

    return convert(copy.copy(expr))
