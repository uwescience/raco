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
    def evaluate(self, _tuple, scheme, state=None):
        return abs(self.input.evaluate(_tuple, scheme, state))


class CEIL(UnaryFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.ceil(self.input.evaluate(_tuple, scheme, state))


class COS(UnaryFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.cos(self.input.evaluate(_tuple, scheme, state))


class FLOOR(UnaryFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.floor(self.input.evaluate(_tuple, scheme, state))


class LOG(UnaryFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.log(self.input.evaluate(_tuple, scheme, state))


class SIN(UnaryFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.sin(self.input.evaluate(_tuple, scheme, state))


class SQRT(UnaryFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.sqrt(self.input.evaluate(_tuple, scheme, state))


class TAN(UnaryFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.tan(self.input.evaluate(_tuple, scheme, state))


class POW(BinaryFunction):
    literals = ['POW']

    def evaluate(self, _tuple, scheme, state=None):
        return pow(self.left.evaluate(_tuple, scheme, state),
                   self.right.evaluate(_tuple, scheme, state))
