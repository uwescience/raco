import abc
from abc import abstractmethod
from raco.expression.function import UnaryFunction, \
    BinaryFunction, NaryFunction


class ExpressionVisitor(object):
    # TODO: make this more complete for kinds of expressions

    __metaclass__ = abc.ABCMeta

    def visit(self, expr):
        # use expr to dispatch to appropriate visit_* method

        # special case names for things we expect are not useful to
        # differentiate further
        # E.g., instead of visit_SQRT, do visit_UnaryFunction
        if isinstance(expr, UnaryFunction):
            typename = UnaryFunction.__name__
        elif isinstance(expr, NaryFunction):
            typename = NaryFunction.__name__
        else:
            # get most names from the expression's type
            typename = type(expr).__name__

        dispatchTo = getattr(self, "visit_%s" % (typename,))
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

    @abstractmethod
    def visit_NamedStateAttributeRef(self, attr):
        return

    @abc.abstractmethod
    def visit_StringLiteral(self, stringLiteral):
        return

    @abc.abstractmethod
    def visit_NumericLiteral(self, numericLiteral):
        return

    @abc.abstractmethod
    def visit_DIVIDE(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_PLUS(self, binaryExpr):
        return

    @abstractmethod
    def visit_MINUS(self, binaryExpr):
        return

    @abstractmethod
    def visit_IDIVIDE(self, binaryExpr):
        return

    @abstractmethod
    def visit_MOD(self, binaryExpr):
        return

    @abstractmethod
    def visit_TIMES(self, binaryExpr):
        return

    @abstractmethod
    def visit_NEG(self, unaryExpr):
        return

    @abstractmethod
    def visit_Case(self, caseExpr):
        return

    @abstractmethod
    def visit_UnaryFunction(self, expr):
        return

    @abstractmethod
    def visit_BinaryFunction(self, expr):
        return

    @abstractmethod
    def visit_NaryFunction(self, expr):
        return

    @abstractmethod
    def visit_CAST(self, expr):
        return

    @abstractmethod
    def visit_LIKE(self, binaryExpr):
        return


class SimpleExpressionVisitor(ExpressionVisitor):

    @abstractmethod
    def visit_unary(self, unaryexpr):
        pass

    @abstractmethod
    def visit_binary(self, binaryexpr):
        pass

    @abstractmethod
    def visit_zeroary(self, zeroaryexpr):
        pass

    @abstractmethod
    def visit_literal(self, literalexpr):
        pass

    @abstractmethod
    def visit_nary(self, naryexpr):
        pass

    @abstractmethod
    def visit_attr(self, attr):
        pass

    def visit_NOT(self, unaryExpr):
        self.visit_unary(unaryExpr)

    def visit_AND(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_OR(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_EQ(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_NEQ(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_GT(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_LT(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_GTEQ(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_LTEQ(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_NamedAttributeRef(self, named):
        self.visit_attr(named)

    def visit_UnnamedAttributeRef(self, unnamed):
        self.visit_attr(unnamed)

    def visit_NamedStateAttributeRef(self, attr):
        self.visit_attr(attr)

    def visit_StringLiteral(self, stringLiteral):
        self.visit_literal(stringLiteral)

    def visit_NumericLiteral(self, numericLiteral):
        self.visit_literal(numericLiteral)

    def visit_DIVIDE(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_PLUS(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_MINUS(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_IDIVIDE(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_MOD(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_TIMES(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_NEG(self, unaryExpr):
        self.visit_unary(unaryExpr)

    def visit_LIKE(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_UnaryFunction(self, expr):
        self.visit_unary(expr)

    def visit_BinaryFunction(self, expr):
        self.visit_binary(expr)

    def visit_NaryFunction(self, expr):
        self.visit_nary(expr)

    def visit_CAST(self, expr):
        self.visit_unary(expr)
