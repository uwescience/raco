"""
Utility functions for use in Raco expressions
"""

from .expression import *
from .aggregate import *
from .boolean import *
from .function import *

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
    import raco.expression as expr
    """Return a list of all classes in the module"""
    return [c for c in expr.__dict__.values() if not hasattr(c, "__class__")]

def aggregate_functions():
    """Return all the classes that can be used to construct an aggregate expression"""
    allclasses = all_classes()
    opclasses = [opclass for opclass in allclasses
                     if issubclass(opclass, AggregateExpression)
                     and opclass is not AggregateExpression]

    return opclasses

def binary_ops():
    """Return a list of all classes used to construct arithmetic, like PLUS, DIVIDE, etc."""
    allclasses = all_classes()
    opclasses = [opclass for opclass in allclasses
                     if issubclass(opclass, BinaryOperator)
                     and opclass is not BinaryOperator
                     and opclass is not BinaryBooleanOperator
                     and opclass is not BinaryComparisonOperator
                     and opclass is not BinaryFunction
                     and not issubclass(opclass,AggregateExpression)]
    return opclasses

def isaggregate(expr):
    return any(expr.postorder(lambda x: isinstance(x, AggregateExpression)))
