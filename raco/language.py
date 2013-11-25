from raco import expression

from abc import ABCMeta, abstractmethod

class Language(object):
    __metaclass__ = ABCMeta

    # By default, reuse scans
    reusescans = True

    @staticmethod
    def preamble(query=None, plan=None):
        return  ""

    @staticmethod
    def postamble(query=None, plan=None):
        return  ""

    @staticmethod
    def initialize(resultsym):
        return  ""

    @staticmethod
    def finalize(resultsym, body=""):
        return ""

    @classmethod
    def compile_stringliteral(cls, value):
        return '"%s"' % value

    @staticmethod
    def log(txt):
        """Emit code that will generate a log message at runtime. Defaults to nothing."""
        return ""

    @classmethod
    def compile_numericliteral(cls, value):
        return '%s' % value

    @classmethod
    def compile_attribute(cls, attr):
        return attr.compile()

    @classmethod
    def conjunction(cls, *args):
        return cls.boolean_combine(args, operator="and")


    @classmethod
    def disjunction(cls, *args):
        return cls.boolean_combine(args, operator="or")

    @classmethod
    def unnamed(cls, condition, sch):
        """
    Replace column names with positions
    """
        if isinstance(condition, expression.BinaryBooleanOperator):
            condition.left = Language.unnamed(condition.left, sch)
            condition.right = Language.unnamed(condition.right, sch)
            result = condition

        elif isinstance(condition, expression.UnaryBooleanOperator):
            condition.input = Language.unnamed(condition.input, sch)
            result = condition

        elif isinstance(condition, expression.NamedAttributeRef):
            pos = sch.getPosition(condition.name)
            result = expression.UnnamedAttributeRef(pos)
        elif isinstance(condition, expression.UnnamedAttributeRef):
            result = condition
        else:
            # do nothing; it's a literal or something custom
            result = condition

        return result

    @classmethod
    def compile_boolean(cls, expr):
        """Compile a boolean condition into the target language"""
        if isinstance(expr, expression.UnaryBooleanOperator):
            input = cls.compile_boolean(expr.input)
            if isinstance(expr, expression.NOT):
                return cls.negation(input)
        if isinstance(expr, expression.BinaryBooleanOperator):
            left, right = cls.compile_boolean(expr.left), cls.compile_boolean(expr.right)
            if isinstance(expr, expression.AND):
                return cls.conjunction(left, right)
            if isinstance(expr, expression.OR):
                return cls.disjunction(left, right)
            if isinstance(expr, expression.EQ):
                return cls.boolean_combine([left, right], operator="==")
            if isinstance(expr, expression.NEQ):
                return cls.boolean_combine([left, right], operator="!=")
            if isinstance(expr, expression.GT):
                return cls.boolean_combine([left, right], operator=">")
            if isinstance(expr, expression.LT):
                return cls.boolean_combine([left, right], operator="<")
            if isinstance(expr, expression.GTEQ):
                return cls.boolean_combine([left, right], operator=">=")
            if isinstance(expr, expression.LTEQ):
                return cls.boolean_combine([left, right], operator="<=")

        elif isinstance(expr, expression.NamedAttributeRef):
            return cls.compile_attribute(expr)

        elif isinstance(expr, expression.StringLiteral):
            return cls.compile_stringliteral(expr.value)

        elif isinstance(expr, expression.NumericLiteral):
            return cls.compile_numericliteral(expr.value)

        elif isinstance(expr, expression.UnnamedAttributeRef):
            return cls.compile_attribute(expr)

        else:
            return expr
            #raise ValueError("Unknown class in boolean expression: %s (value is %s)" % (expr.__class__,expr))

    @abstractmethod
    def boolean_combine(cls, args, operator="and"):
        """Combine the given arguments using the specified infix operator"""

# import everything from each language
from raco.pythonlang import PythonAlgebra
from raco.pseudocodelang import PseudoCodeAlgebra
#from raco.clang import CCAlgebra
from raco.myrialang import MyriaAlgebra
