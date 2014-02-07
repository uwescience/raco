"""
Boolean operators for use in Raco expression trees
"""

from .expression import Expression, UnaryOperator, BinaryOperator, \
        AttributeRef, NumericLiteral

import abc

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



class BooleanExprVisitor(object):
    __metaclass__ = abc.ABCMeta

    def visit(self, expr):
        # use expr to dispatch to appropriate visit_* method
        typename = type(expr).__name__
        dispatchTo = getattr(self, "visit_%s"%(typename))
        return dispatchTo(expr)

    @abc.abstractmethod
    def visit_NOT(self, unaryExpr):
        return

    @abc.abstractmethod
    def visit_AND(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_OR(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_EQ(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_NEQ(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_NEQ(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_GT(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_LT(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_GTEQ(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_LTEQ(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_NamedAttributeRef(self, named):
        return

    @abc.abstractmethod
    def visit_UnnamedAttributeRef(self, unnamed):
        return

    @abc.abstractmethod
    def visit_StringLiteral(self, stringLiteral):
        return

    @abc.abstractmethod
    def visit_NumericLiteral(self, numericLiteral):
        return

import sympy
class SympyBuildVisitor(BooleanExprVisitor):
    def __init__(self):
        self.stack = []
        self.symbolmappings = {}
        self.i = 0
        self.valid = True

    def __newsym__(self):
        newsym = sympy.symbols("x%s" % (self.i))
        self.i += 1
        return newsym

    def __visit_BinaryBooleanOperator__(self, binaryexpr):
        right = self.stack.pop()
        left = self.stack.pop()
        return left, right

    def visit_NOT(self, unaryExpr):
        inputsym = self.stack.pop()
        self.stack.append(~inputsym)

    def visit_AND(self, binaryExpr):
        leftsym, rightsym = self.__visit_BinaryBooleanOperator__(binaryExpr)
        self.stack.append(leftsym & rightsym)

    def visit_OR(self, binaryExpr):
        leftsym, rightsym = self.__visit_BinaryBooleanOperator__(binaryExpr)
        self.stack.append(leftsym | rightsym)

    def visit_GT(self, binaryExpr):
        leftsym, rightsym = self.__visit_BinaryBooleanOperator__(binaryExpr)
        self.stack.append(sympy.Gt(leftsym, rightsym))

    def visit_GTEQ(self, binaryExpr):
        leftsym, rightsym = self.__visit_BinaryBooleanOperator__(binaryExpr)
        self.stack.append(sympy.Ge(leftsym, rightsym))

    def visit_LT(self, binaryExpr):
        leftsym, rightsym = self.__visit_BinaryBooleanOperator__(binaryExpr)
        self.stack.append(sympy.Lt(leftsym, rightsym))

    def visit_LTEQ(self, binaryExpr):
        leftsym, rightsym = self.__visit_BinaryBooleanOperator__(binaryExpr)
        self.stack.append(sympy.Le(leftsym, rightsym))

    def visit_NEQ(self, binaryExpr):
        leftsym, rightsym = self.__visit_BinaryBooleanOperator__(binaryExpr)
        self.stack.append(sympy.Ne(leftsym, rightsym))

    def visit_EQ(self, binaryExpr):
        leftsym, rightsym = self.__visit_BinaryBooleanOperator__(binaryExpr)
        self.stack.append(sympy.Eq(leftsym, rightsym))

    def visit_UnnamedAttributeRef(self, unnamed):
        newsym = self.__newsym__()
        self.symbolmappings[newsym] = unnamed
        self.stack.append(newsym)

    def visit_NamedAttributeRef(self, named):
        newsym = self.__newsym__()
        self.symbolmappings[newsym] = named
        self.stack.append(newsym)

    def visit_NumericLiteral(self, numericLiteral):
        self.stack.append(sympy.Integer(numericLiteral.value))

    def visit_StringLiteral(self, stringLiteral):
        # no support for simplification containing strings
        self.valid = False

        # just append a dummy number so the visits can finish
        self.stack.append(sympy.Integer(0))


