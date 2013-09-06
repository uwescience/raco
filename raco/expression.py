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


class ZeroaryOperator(Expression):
  def __init__(self):

  def __eq__(self, other):
    return self.__class__ == other.__class__ 

  def __str__(self):
    return "%s" % (self.opstr(),)

  def __repr__(self):
    return self.__str__()


class UnaryOperator(Expression):
  def __init__(self, input):
    self.input = input

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.input == other.input

  def __str__(self):
    return "%s%s" % (self.opstr(), self.input)

  def __repr__(self):
    return self.__str__()

class BinaryOperator(Expression):
  def __init__(self, left, right):
    self.left = left
    self.right = right

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.left == other.left and self.right == other.right

  def __str__(self):
    return "(%s %s %s)" % (self.left, self.opstr(), self.right)

  def __repr__(self):
    return self.__str__()



class Literal:
  def __init__(self, value):
    self.value = value

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.value == other.value

  def vars(self):
    return []

  def __repr__(self):
    return str(self.value)

  def typeof(self):
    # TODO: DANGEROUS
    return type(self.value)

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

class NaryOperator(Expression):
  pass

class UDF(NaryOperator):
  pass

class PLUS(BinaryOperator):
  literals = ["+"]

class MINUS(BinaryOperator):
  literals = ["-"]

class DIVIDE(BinaryOperator):
  literals = ["/"]

class TIMES(BinaryOperator):
  literals = ["*"]

class NEG(UnaryOperator):
  literals = ["-"]

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
  pass

class MAX(AggregateExpression,UnaryFunction):
  pass

class MIN(AggregateExpression,UnaryFunction):
  pass

class COUNT(AggregateExpression,ZeroaryOperator):
  pass

class SUM(AggregateExpression,UnaryFunction):
  pass

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

class AND(BinaryBooleanOperator):
  literals = ["and", "AND"]

class OR(BinaryBooleanOperator):
  literals = ["or", "OR"]

class EQ(BinaryComparisonOperator):
  literals = ["=", "=="]

class LT(BinaryComparisonOperator):
  literals = ["<", "lt"]

class GT(BinaryComparisonOperator):
  literals = [">", "gt"]

class GTEQ(BinaryComparisonOperator):
  literals = [">=", "gteq", "gte"]

class LTEQ(BinaryComparisonOperator):
  literals = ["<=", "lteq", "lte"]

class NEQ(BinaryComparisonOperator):
  literals = ["!=", "neq", "ne"]

reverse = {
  NEQ:NEQ,
  EQ:EQ,
  GTEQ:LTEQ,
  LTEQ:GTEQ,
  GT:LT,
  LT:GT
}

def toUnnamed(ref, scheme):
  """Convert a reference to the unnamed perspective"""
  if issubclass(ref, UnnamedAttributeRef):
    return ref
  elif issubclass(ref, NamedAttributeRef):
    return UnnamedAttributeRef(scheme.getpos(ref.name))
  else:
    raise TypeError("Unknown value reference %s.  Expected a position reference or an attribute reference.")

def toNamed(ref, scheme):
  """Convert a reference to the named perspective"""
  if issubclass(ref, UnnamedAttributeRef):
    attrname = scheme[ref.position][0]
    return NamedAttributeRef(attrname)
  elif issubclass(ref, NamedAttributeRef):
    return ref
  else:
    raise TypeError("Unknown value reference %s.  Expected a position reference or an attribute reference.")
