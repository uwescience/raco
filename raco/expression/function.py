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

    def typeof(self, scheme, state_scheme):
        return "LONG_TYPE"


class UnaryLongFunction(UnaryFunction):
    def typeof(self, scheme, state_scheme):
        input_type = self.input.typeof(scheme, state_scheme)
        check_is_numeric(input_type)
        return "LONG_TYPE"


class UnaryFloatFunction(UnaryFunction):
    def typeof(self, scheme, state_scheme):
        input_type = self.input.typeof(scheme, state_scheme)
        check_is_numeric(input_type)
        return "DOUBLE_TYPE"


class UnaryTypePreservingFunction(UnaryFunction):
    def typeof(self, scheme, state_scheme):
        input_type = self.input.typeof(scheme, state_scheme)
        check_is_numeric(input_type)
        return input_type


class StringFunction(UnaryFunction):
    def typeof(self, scheme, state_scheme):
        input_type = self.input.typeof(scheme, state_scheme)
        if input_type != "STRING_TYPE":
            raise TypeSafetyViolation("Must be a string for %s" % (
                self.__class__,))
        return "STRING_TYPE"


class ABS(UnaryTypePreservingFunction):    
    def evaluate(self, _tuple, scheme, state=None):
        return abs(self.input.evaluate(_tuple, scheme, state))


class CEIL(UnaryLongFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.ceil(self.input.evaluate(_tuple, scheme, state))


class COS(UnaryFloatFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.cos(self.input.evaluate(_tuple, scheme, state))


class FLOOR(UnaryLongFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.floor(self.input.evaluate(_tuple, scheme, state))


class LOG(UnaryFloatFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.log(self.input.evaluate(_tuple, scheme, state))


class SIN(UnaryFloatFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.sin(self.input.evaluate(_tuple, scheme, state))


class SQRT(UnaryFloatFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.sqrt(self.input.evaluate(_tuple, scheme, state))


class TAN(UnaryFloatFunction):
    def evaluate(self, _tuple, scheme, state=None):
        return math.tan(self.input.evaluate(_tuple, scheme, state))


class POW(BinaryFunction):
    literals = ['POW']

    def evaluate(self, _tuple, scheme, state=None):
        return pow(self.left.evaluate(_tuple, scheme, state),
                   self.right.evaluate(_tuple, scheme, state))

    def typeof(self, scheme, state_scheme):
        lt = self.left.typeof(scheme, state_scheme)
        check_is_numeric(lt)
        rt = self.right.typeof(scheme, state_scheme)
        check_is_numeric(rt)

        return "DOUBLE_TYPE"

class CompareFunction(BinaryFunction):
    def typeof(self, scheme, state_scheme):
        lt = self.left.typeof(scheme, state_scheme)
        rt = self.right.typeof(scheme, state_scheme)
        if lt != rt:
            raise TypeSafetyViolation("Can't compare %s with %s" % (
                lt, rt))
        return lt


class LESSER(CompareFunction):
    literals = ['LESSER']

    def evaluate(self, _tuple, scheme, state=None):
        return min(self.left.evaluate(_tuple, scheme, state),
                   self.right.evaluate(_tuple, scheme, state))


class GREATER(CompareFunction):
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

    def typeof(self, scheme, state_scheme):
        return "STRING_TYPE"


class LEN(StringFunction):
    literals = ["LEN"]

    def evaluate(self, _tuple, scheme, state=None):
        return len(self.input.evaluate(_tuple, scheme, state))
