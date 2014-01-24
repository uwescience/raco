"""
Utility functions for use in Raco expressions
"""

from .expression import BinaryOperator, NamedAttributeRef, UnnamedAttributeRef
from .aggregate import AggregateExpression

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
