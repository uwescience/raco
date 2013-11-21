"""
Boolean operators for use in Raco expression trees
"""

from .expression import Expression, UnaryOperator, BinaryOperator, \
        AttributeRef, NumericLiteral

class BooleanExpression(Expression):
  pass

class UnaryBooleanOperator(UnaryOperator, BooleanExpression):
  pass

class BinaryBooleanOperator(BinaryOperator, BooleanExpression):
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

TAUTOLOGY = EQ(NumericLiteral(1), NumericLiteral(1))

def is_column_comparison(expr, scheme):
  """Return a truthy value if the expression is a comparison between columns.

  The return value is a tuple containing the column indexes, or
  None if the expression is not a simple column comparison.
  """

  if isinstance(expr, EQ) and isinstance(expr.left, AttributeRef) \
     and isinstance(expr.right, AttributeRef):
    return (toUnnamed(expr.left, scheme).position,
            toUnnamed(expr.right, scheme).position)
  else:
    return None


def extract_conjuncs(sexpr):
  """Return a list of conjunctions from a scalar expression."""

  if isinstance(sexpr, AND):
      left = extract_conjuncs(sexpr.left)
      right = extract_conjuncs(sexpr.right)
      return left + right
  else:
    return [sexpr]

from .util import toUnnamed
