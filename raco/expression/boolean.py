"""
Boolean operators for use in Raco expression trees
"""

from .expression import Expression, UnaryOperator, BinaryOperator, \
    AttributeRef, NumericLiteral

import abc


class BooleanExpression(Expression):
    def typeof(self, scheme, state_scheme):
        return "BOOLEAN_TYPE"


class UnaryBooleanOperator(BooleanExpression, UnaryOperator):
    pass


class BinaryBooleanOperator(BooleanExpression, BinaryOperator):
    pass


class BinaryComparisonOperator(BinaryBooleanOperator):
    pass


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
