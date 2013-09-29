from utility import emit, Printable
"""
An expression language for datalog: Function calls, arithmetic, simple string functions
"""

class Expression(Printable):
  def typeof(self):
    # By default, we don't know the type
    return None

  def opstr(self):
    if not hasattr(self, "literals"):
      return self.opname()
    else:
      return self.literals[0]

  def evaluate(self, _tuple, scheme):
    '''Evaluate an expression in the context of a given tuple and schema.

    This is used for unit tests written against the fake database.
    '''
    raise NotImplementedError()

  def postorder(self, f):
    yield f(self)

  def apply(self, f):
    """Replace children with the result of a function"""
    pass

class ZeroaryOperator(Expression):
  def __init__(self):
    pass
  def __eq__(self, other):
    return self.__class__ == other.__class__ 

  def __hash__(self):
    return hash(self.__class__)

  def __str__(self):
    return "%s" % (self.opstr(),)

  def __repr__(self):
    return self.__str__()

class UnaryOperator(Expression):
  def __init__(self, input):
    self.input = input

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.input == other.input

  def __hash__(self):
    return hash(self.__class__) + hash(self.input)

  def __str__(self):
    return "%s%s" % (self.opstr(), self.input)

  def __repr__(self):
    return self.__str__()

  def postorder(self, f):
    for x in self.input.postorder(f):
      yield x
    yield f(self)

  def apply(self, f):
    self.input = f(self.input)


class BinaryOperator(Expression):
  def __init__(self, left, right):
    self.left = left
    self.right = right

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.left == other.left and self.right == other.right

  def __hash__(self):
    return hash(self.__class__) + hash(self.left) + hash(self.right)

  def __str__(self):
    return "(%s %s %s)" % (self.left, self.opstr(), self.right)

  def __repr__(self):
    return self.__str__()

  def postorder(self, f):
    for x in self.left.postorder(f):
      yield x
    for x in self.right.postorder(f):
      yield x
    yield f(self)

  def apply(self, f):
    self.left = f(self.left)
    self.right = f(self.right)

class Literal(ZeroaryOperator):
  def __init__(self, value):
    self.value = value

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.value == other.value

  def __hash__(self):
    return hash(self.__class__) + hash(self.value)

  def vars(self):
    return []

  def __repr__(self):
    return str(self.value)

  def typeof(self):
    # TODO: DANGEROUS
    return type(self.value)

  def evaluate(self, _tuple, scheme):
    return self.value

  def apply(self, f):
    pass

class StringLiteral(Literal):
  pass

class NumericLiteral(Literal):
  pass

class AttributeRef(Expression):
  pass

class NamedAttributeRef(AttributeRef):
  def __init__(self, attributename):
    self.name = attributename

  def __repr__(self):
    return "%s" % (self.name)

  def __str__(self):
    return "%s" % (self.name)

  def evaluate(self, _tuple, scheme):
    pos = scheme.getPosition(self.name)
    return _tuple[pos]

class UnnamedAttributeRef(AttributeRef):
  def __init__(self, position):
    self.position = position

  def __repr__(self):
    return "$%s" % (self.position)

  def __str__(self):
    return "$%s" % (self.position)

  # TODO: These are artifacts of a messy way of handling attribute references
  # Hopefully will go away.
  def leftoffset(self, offset):
    """Add an offset to this positional reference.  Used when building a plan from a set of joins"""
    self.position = self.position + offset

  def rightoffset(self, offset):
    """Add an offset to this positional reference.  Used when building a plan from a set of joins"""
    self.position = self.position + offset

  def evaluate(self, _tuple, scheme):
    return _tuple[self.position]

class DottedAttributeRef(AttributeRef):
  def __init__(self, relation_name, field):
    """Initializse a DottedAttributeRef.

    relation_name is a string that refers to a relation.
    field refers to a column; its value is either a string or an integer.
    """
    self.relation_name = relation_name
    self.field = field

  def __repr__(self):
    return "%s.%s" % (self.relation_name, str(self.field))

  def __str__(self):
    return "%s.%s" % (self.relation_name, str(self.field))

  def evaluate(self, _tuple, scheme):
    """Panic on attempted evaluation.

    DottedAttributeRefs are compiled away into UnnamedAttributeRefs before
    evaluation.  See unpack_from.py.
    """
    raise NotImplementedError()

class NaryOperator(Expression):
  pass

class UDF(NaryOperator):
  pass

class PLUS(BinaryOperator):
  literals = ["+"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) +
            self.right.evaluate(_tuple, scheme))

class MINUS(BinaryOperator):
  literals = ["-"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) -
            self.right.evaluate(_tuple, scheme))

class DIVIDE(BinaryOperator):
  literals = ["/"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) /
            self.right.evaluate(_tuple, scheme))


class TIMES(BinaryOperator):
  literals = ["*"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) *
            self.right.evaluate(_tuple, scheme))


class NEG(UnaryOperator):
  literals = ["-"]

  def evaluate(self, _tuple, scheme):
    return -1 * self.input.evaluate(_tuple, scheme)

class UnaryFunction(UnaryOperator):
  def __str__(self):
    return "%s(%s)" % (self.__class__.__name__, self.input)

class ABS(UnaryFunction):
  pass

class SQRT(UnaryFunction):
  pass

class LN(UnaryFunction):
  pass

class LOG(UnaryFunction):
  pass

class SIN(UnaryFunction):
  pass

class COS(UnaryFunction):
  pass

class TAN(UnaryFunction):
  pass

class FLOOR(UnaryFunction):
  pass

class CEIL(UnaryFunction):
  pass

class AggregateExpression(Expression):
  def evaluate(self, _tuple, scheme):
    """Stub evaluate function for aggregate expressions.

    Aggregate functions do not evaluate individual tuples; rather they
    operate on collections of tuples in the evaluate_aggregate function.
    We return a dummy string so that all tuples containing this aggregate
    hash to the same value.
    """
    return self.opname()

  def evaluate_aggregate(self, tuple_iterator, scheme):
    """Evaluate an aggregate over a bag of tuples"""
    raise NotImplementedError()

class MAX(UnaryFunction,AggregateExpression):
  def evaluate_aggregate(self, tuple_iterator, scheme):
    inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
    return max(inputs)

class MIN(UnaryFunction,AggregateExpression):
  def evaluate_aggregate(self, tuple_iterator, scheme):
    inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
    return min(inputs)

class COUNT(UnaryFunction,AggregateExpression):
  def evaluate_aggregate(self, tuple_iterator, scheme):
    inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
    count = 0
    for t in inputs:
      if t is not None:
        count += 1
    return count

class SUM(UnaryFunction,AggregateExpression):
  def evaluate_aggregate(self, tuple_iterator, scheme):
    inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)

    sum = 0
    for t in inputs:
      if t is not None:
        sum += t
    return sum

class BooleanExpression(Printable):
  pass

class UnaryBooleanOperator(UnaryOperator,BooleanExpression):
  pass

class BinaryBooleanOperator(BinaryOperator,BooleanExpression):
  pass

class BinaryComparisonOperator(BinaryBooleanOperator):
  pass

class NOT(UnaryBooleanOperator):
  literals = ["not", "NOT", "-"]

  def evaluate(self, _tuple, scheme):
    return not self.input.evaluate(_tuple, scheme)

class AND(BinaryBooleanOperator):
  literals = ["and", "AND"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) and
            self.right.evaluate(_tuple, scheme))

class OR(BinaryBooleanOperator):
  literals = ["or", "OR"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) or
            self.right.evaluate(_tuple, scheme))

class EQ(BinaryComparisonOperator):
  literals = ["=", "=="]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) ==
            self.right.evaluate(_tuple, scheme))

class LT(BinaryComparisonOperator):
  literals = ["<", "lt"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) <
            self.right.evaluate(_tuple, scheme))


class GT(BinaryComparisonOperator):
  literals = [">", "gt"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) >
            self.right.evaluate(_tuple, scheme))


class GTEQ(BinaryComparisonOperator):
  literals = [">=", "gteq", "gte"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) >=
            self.right.evaluate(_tuple, scheme))


class LTEQ(BinaryComparisonOperator):
  literals = ["<=", "lteq", "lte"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) <=
            self.right.evaluate(_tuple, scheme))


class NEQ(BinaryComparisonOperator):
  literals = ["!=", "neq", "ne"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) !=
            self.right.evaluate(_tuple, scheme))

reverse = {
  NEQ:NEQ,
  EQ:EQ,
  GTEQ:LTEQ,
  LTEQ:GTEQ,
  GT:LT,
  LT:GT
}

class Unbox(ZeroaryOperator):
  def __init__(self, table, field):
    self.table = table
    self.field = field

  def evaluate(self, _tuple, scheme):
    """Raise an error on attempted evaluation.

    Unbox should never be "evaluated" in the usual sense.  Rather it should
    be replaced by a cross-product with a single-element table.  This operator
    is just a placeholder.
    """
    raise NotImplementedError()

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
                   and not issubclass(opclass,AggregateExpression)]
  return opclasses

def isaggregate(expr):
  return any(expr.postorder(lambda x: isinstance(x, AggregateExpression)))
