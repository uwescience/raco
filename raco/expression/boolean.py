"""
Boolean operators for use in Raco expression trees
"""

from .expression import Expression, UnaryOperator, BinaryOperator, \
    AttributeRef, NumericLiteral, check_type, TypeSafetyViolation

import raco.types

import abc


class BooleanExpression(Expression):
    pass


class UnaryBooleanOperator(BooleanExpression, UnaryOperator):
    def typeof(self, scheme, state_scheme):
        lt = self.input.typeof(scheme, state_scheme)
        check_type(lt, raco.types.BOOLEAN_TYPE)
        return raco.types.BOOLEAN_TYPE


class BinaryBooleanOperator(BooleanExpression, BinaryOperator):
    def typeof(self, scheme, state_scheme):
        lt = self.left.typeof(scheme, state_scheme)
        check_type(lt, raco.types.BOOLEAN_TYPE)
        rt = self.right.typeof(scheme, state_scheme)
        check_type(rt, raco.types.BOOLEAN_TYPE)
        return raco.types.BOOLEAN_TYPE


class BinaryComparisonOperator(BinaryBooleanOperator):
    def typeof(self, scheme, state_scheme):
        lt = self.left.typeof(scheme, state_scheme)
        rt = self.right.typeof(scheme, state_scheme)
        if lt == rt:
            return raco.types.BOOLEAN_TYPE
        if lt in raco.types.NUMERIC_TYPES and rt in raco.types.NUMERIC_TYPES:
            return raco.types.BOOLEAN_TYPE
        else:
            raise TypeSafetyViolation("Can't compare %s and %s", (lt, rt))


class NOT(UnaryBooleanOperator):
    literals = ["not", "NOT", "-"]

    def evaluate(self, _tuple, scheme, state=None):
        return not self.input.evaluate(_tuple, scheme, state)


class AND(BinaryBooleanOperator):
    literals = ["and", "AND"]

    def evaluate(self, _tuple, scheme, state=None):
        return (self.left.evaluate(_tuple, scheme, state) and
                self.right.evaluate(_tuple, scheme, state))


class OR(BinaryBooleanOperator):
    literals = ["or", "OR"]

    def evaluate(self, _tuple, scheme, state=None):
        return (self.left.evaluate(_tuple, scheme, state) or
                self.right.evaluate(_tuple, scheme, state))


class EQ(BinaryComparisonOperator):
    literals = ["=", "=="]

    def evaluate(self, _tuple, scheme, state=None):
        return (self.left.evaluate(_tuple, scheme, state) ==
                self.right.evaluate(_tuple, scheme, state))


class LT(BinaryComparisonOperator):
    literals = ["<", "lt"]

    def evaluate(self, _tuple, scheme, state=None):
        return (self.left.evaluate(_tuple, scheme, state) <
                self.right.evaluate(_tuple, scheme, state))


class GT(BinaryComparisonOperator):
    literals = [">", "gt"]

    def evaluate(self, _tuple, scheme, state=None):
        return (self.left.evaluate(_tuple, scheme, state) >
                self.right.evaluate(_tuple, scheme, state))


class GTEQ(BinaryComparisonOperator):
    literals = [">=", "gteq", "gte"]

    def evaluate(self, _tuple, scheme, state=None):
        return (self.left.evaluate(_tuple, scheme, state) >=
                self.right.evaluate(_tuple, scheme, state))


class LTEQ(BinaryComparisonOperator):
    literals = ["<=", "lteq", "lte"]

    def evaluate(self, _tuple, scheme, state=None):
        return (self.left.evaluate(_tuple, scheme, state) <=
                self.right.evaluate(_tuple, scheme, state))


class NEQ(BinaryComparisonOperator):
    literals = ["!=", "neq", "ne"]

    def evaluate(self, _tuple, scheme, state=None):
        return (self.left.evaluate(_tuple, scheme, state) !=
                self.right.evaluate(_tuple, scheme, state))


class LIKE(BinaryComparisonOperator):
    literals = ["like"]

    def evaluate(self, _tuple, scheme, state=None):
        raise NotImplementedError("TODO: use regex")

reverse = {
    NEQ: NEQ,
    EQ: EQ,
    GTEQ: LTEQ,
    LTEQ: GTEQ,
    GT: LT,
    LT: GT
}

TAUTOLOGY = EQ(NumericLiteral(1), NumericLiteral(1))


def extract_conjuncs(sexpr):
    """Return a list of conjunctions from a scalar expression."""

    if isinstance(sexpr, AND):
        left = extract_conjuncs(sexpr.left)
        right = extract_conjuncs(sexpr.right)
        return left + right
    else:
        return [sexpr]

from .util import toUnnamed
