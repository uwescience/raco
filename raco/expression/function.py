"""
Functions (unary and binary) for use in Raco.
"""

from .expression import UnaryOperator, BinaryOperator
import math

class UnaryFunction(UnaryOperator):
  def __str__(self):
    return "%s(%s)" % (self.__class__.__name__, self.input)

class BinaryFunction(BinaryOperator):
  def __str__(self):
    return "%s(%s, %s)" % (self.__class__.__name__, self.left, self.right)


class ABS(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return abs(self.input.evaluate(_tuple, scheme))

class CEIL(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.ceil(self.input.evaluate(_tuple, scheme))

class COS(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.cos(self.input.evaluate(_tuple, scheme))

class FLOOR(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.floor(self.input.evaluate(_tuple, scheme))

class LOG(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.log(self.input.evaluate(_tuple, scheme))

class SIN(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.sin(self.input.evaluate(_tuple, scheme))

class SQRT(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.sqrt(self.input.evaluate(_tuple, scheme))

class TAN(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.tan(self.input.evaluate(_tuple, scheme))

class POW(BinaryFunction):
  literals = ['POW']
  def evaluate(self, _tuple, scheme):
    return pow(self.left.evaluate(_tuple, scheme),
               self.right.evaluate(_tuple, scheme))
