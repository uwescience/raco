"""
Functions (unary and binary) for use in Raco.
"""

import math

from .expression import *


class UnaryFunction(UnaryOperator):
    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.input)


class BinaryFunction(BinaryOperator):
    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.left, self.right)


class NaryFunction(NaryOperator):
    def __str__(self):
        return "%s(%s)" % \
            (self.__class__.__name__,
             ",".join([str(op) for op in self.operands]))


class WORKERID(ZeroaryOperator):
    def __str__(self):
        return "%s" % self.__class__.__name__

    def evaluate(self, _tuple, scheme, state=None):
        return 0


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


class LESSER(BinaryFunction):
    literals = ['LESSER']

    def evaluate(self, _tuple, scheme, state=None):
        return min(self.left.evaluate(_tuple, scheme, state),
                   self.right.evaluate(_tuple, scheme, state))


class GREATER(BinaryFunction):
    literals = ['GREATER']

    def evaluate(self, _tuple, scheme, state=None):
        return max(self.left.evaluate(_tuple, scheme, state),
                   self.right.evaluate(_tuple, scheme, state))


class SUBSTR(NaryFunction):
    literals = ["SUBSTR"]

    def evaluate(self, _tuple, scheme, state=None):
        inputStr = self.operands[0].evaluate(_tuple, scheme, state)
        beginIdx = self.operands[1].evaluate(_tuple, scheme, state)
        endIdx = self.operands[2].evaluate(_tuple, scheme, state)
        return inputStr[beginIdx:endIdx]
