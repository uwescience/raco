from abc import ABCMeta, abstractmethod

import logging
from raco.expression.visitor import ExpressionVisitor

LOG = logging.getLogger(__name__)


class Algebra(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def opt_rules(self, **kwargs):
        raise NotImplementedError("{op}.opt_rules()".format(op=type(self)))


class Language(object):
    __metaclass__ = ABCMeta

    EQ = "=="
    NEQ = "!="
    PLUS = "+"
    MINUS = "-"
    DIVIDE = "/"
    TIMES = "*"
    IDIVIDE = "//"
    GT = ">"
    LT = "<"
    GTEQ = ">="
    LTEQ = "<="
    MOD = "%"
    LIKE = "like"

    # By default, reuse scans
    reusescans = True

    @staticmethod
    def preamble(query=None, plan=None):
        return ""

    @staticmethod
    def postamble(query=None, plan=None):
        return ""

    @staticmethod
    def body(compileResult):
        return compileResult

    @classmethod
    def compile_stringliteral(cls, value):
        return '"%s"' % value

    @staticmethod
    def log(txt):
        """Emit code that will generate a log message at runtime. Defaults to
        nothing."""
        return ""

    @classmethod
    def compile_numericliteral(cls, value):
        return '%s' % value

    @classmethod
    def compile_attribute(cls, attr, **kwargs):
        return attr.compile()

    @classmethod
    def conjunction(cls, *args):
        return cls.expression_combine(args, operator="and")

    @classmethod
    def disjunction(cls, *args):
        return cls.expression_combine(args, operator="or")

    @classmethod
    def compile_expression(cls, expr, **kwargs):
        compilevisitor = CompileExpressionVisitor(cls, **kwargs)
        expr.accept(compilevisitor)
        return compilevisitor.getresult()

    @classmethod
    def expression_combine(cls, args, operator="and"):
        """Combine the given arguments using the specified infix operator"""
        delim = " %s " % operator
        return delim.join(args)


class CompileExpressionVisitor(ExpressionVisitor):

    def __init__(self, language, **kwargs):
        self.language = language
        self.combine = language.expression_combine
        self.stack = []
        self.kwargs = kwargs

    def getresult(self):
        assert len(self.stack) == 1, \
            "stack is size {0} != 1".format(len(self.stack))
        return self.stack.pop()

    def __visit_BinaryOperator__(self, binaryexpr):
        right = self.stack.pop()
        left = self.stack.pop()
        return left, right

    def appendbinop(self, binaryexpr, languageop):
        left, right = self.__visit_BinaryOperator__(binaryexpr)
        val = self.combine([left, right], operator=languageop)
        self.stack.append(val)

    def visit_NOT(self, unaryexpr):
        inputexpr = self.stack.pop()
        self.stack.append(self.language.negation(inputexpr))

    def visit_AND(self, binaryexpr):
        left, right = self.__visit_BinaryOperator__(binaryexpr)
        self.stack.append(self.language.conjunction(left, right))

    def visit_OR(self, binaryexpr):
        left, right = self.__visit_BinaryOperator__(binaryexpr)
        self.stack.append(self.language.disjunction(left, right))

    def visit_EQ(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.EQ)

    def visit_NEQ(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.NEQ)

    def visit_GT(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.GT)

    def visit_LT(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.LT)

    def visit_GTEQ(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.GTEQ)

    def visit_LTEQ(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.LTEQ)

    def visit_LIKE(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.LIKE)

    def visit_NamedAttributeRef(self, named):
        self.stack.append(
            self.language.compile_attribute(
                named,
                **self.kwargs))

    def visit_UnnamedAttributeRef(self, unnamed):
        LOG.debug("expr %s is UnnamedAttributeRef", unnamed)
        self.stack.append(
            self.language.compile_attribute(
                unnamed,
                **self.kwargs))

    def visit_NumericLiteral(self, numericliteral):
        self.stack.append(self.language.compile_numericliteral(numericliteral))

    def visit_StringLiteral(self, stringliteral):
        self.stack.append(self.language.compile_stringliteral(stringliteral))

    def visit_DIVIDE(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.DIVIDE)

    def visit_PLUS(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.PLUS)

    def visit_MINUS(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.MINUS)

    def visit_IDIVIDE(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.IDIVIDE)

    def visit_MOD(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.MOD)

    def visit_TIMES(self, binaryexpr):
        self.appendbinop(binaryexpr, self.language.TIMES)

    def visit_NEG(self, unaryexpr):
        inputexpr = self.stack.pop()
        self.stack.append(self.language.negative(inputexpr))

    def visit_Case(self, caseexpr):
        if caseexpr.else_expr is not None:
            else_compiled = self.stack.pop()

        when_compiled = []
        for _ in range(len(caseexpr.when_tuples)):
            thenexpr, ifexpr = self.stack.pop(), self.stack.pop()
            when_compiled.insert(0, (ifexpr, thenexpr))

        self.stack.append(
            self.language.conditional(
                when_compiled,
                else_compiled))

    def visit_NamedStateAttributeRef(self, attr):
        self.stack.append(self.language.compile_attribute(attr, **self.kwargs))

    def visit_UnaryFunction(self, expr):
        inputexpr = self.stack.pop()
        self.stack.append(
            self.language.function_call(
                type(expr).__name__,
                inputexpr))

    def visit_BinaryFunction(self, expr):
        left, right = self.__visit_BinaryOperator__(expr)
        self.stack.append(
            self.language.function_call(
                type(expr).__name__,
                left,
                right))

    def visit_CustomBinaryFunction(self, expr):
        left, right = self.__visit_BinaryOperator__(expr)
        self.stack.append(
            self.language.function_call(
                expr.name,
                left,
                right, custom=True))

    def visit_NaryFunction(self, expr):
        arglist = []
        for _ in range(len(expr.operands)):
            arglist.insert(0, self.stack.pop())
        self.stack.append(
            self.language.function_call(
                type(expr).__name__,
                *arglist))

    def visit_CustomZeroaryFunction(self, expr):
        self.stack.append(
            self.language.function_call(expr.name, custom=True))

    def visit_CAST(self, expr):
        inputexpr = self.stack.pop()
        self.stack.append(self.language.cast(expr._type, inputexpr))
