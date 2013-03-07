import boolean

class Language:
 
  # By default, reuse scans
  reusescans = True

  @staticmethod
  def preamble(query=None,plan=None):
    return  ""

  @staticmethod
  def postamble(query=None,plan=None):
    return  ""

  @staticmethod
  def initialize(resultsym):
    return  ""

  @staticmethod
  def finalize(resultsym,body=""):
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
    if isinstance(condition, boolean.BinaryBooleanOperator):
      condition.left = Language.unnamed(condition.left, sch)
      condition.right = Language.unnamed(condition.right, sch)
      result = condition

    elif isinstance(condition, boolean.UnaryBooleanOperator):
      condition.input = Language.unnamed(condition.input, sch)
      result = condition

    elif isinstance(condition, boolean.Attribute):
      # replace the attribute name with it's position in the relation
      # TODO: This won't work with intermediate results from joins
      # Replace with a generic AttributeReference
      pos = sch.getPosition(condition.name)
      result = boolean.PositionReference(pos)
    elif isinstance(condition, boolean.PositionReference):
      # TODO: Replace with a generic AttributeReference
      result = condition
    else:
      # do nothing; it's a literal or something custom
      result = condition

    return result

  @classmethod
  def compile_boolean(cls, expr):
    """
Compile a boolean condition into the target language
  """
    if isinstance(expr, boolean.UnaryBooleanOperator):
      input = cls.compile_boolean(expr.input)
      if isinstance(expr, boolean.NOT):
        return cls.negation(input)
    if isinstance(expr, boolean.BinaryBooleanOperator):
      left, right = cls.compile_boolean(expr.left), cls.compile_boolean(expr.right)
      if isinstance(expr, boolean.AND):
        return cls.conjunction(left, right)
      if isinstance(expr, boolean.OR):
        return cls.disjunction(left, right)
      if isinstance(expr, boolean.EQ):
        return cls.boolean_combine([left, right], operator="==")
      if isinstance(expr, boolean.NEQ):
        return cls.boolean_combine([left, right], operator="!=")
      if isinstance(expr, boolean.GT):
        return cls.boolean_combine([left, right], operator=">")
      if isinstance(expr, boolean.LT):
        return cls.boolean_combine([left, right], operator="<")
      if isinstance(expr, boolean.GTEQ):
        return cls.boolean_combine([left, right], operator=">=")
      if isinstance(expr, boolean.LTEQ):
        return cls.boolean_combine([left, right], operator="<=")

    elif isinstance(expr, boolean.Attribute):
      return cls.compile_attribute(expr)

    elif isinstance(expr, boolean.StringLiteral):
      return cls.compile_stringliteral(expr.value)
  
    elif isinstance(expr, boolean.NumericLiteral):
      return cls.compile_numericliteral(expr.value)
  
    elif isinstance(expr, boolean.PositionReference):
      return cls.compile_attribute(expr)
  
    else:
      return expr
      #raise ValueError("Unknown class in boolean expression: %s (value is %s)" % (expr.__class__,expr))

# import everything from each language
from pythonlang import *
from pseudocodelang import *
from clang import *
from myrialang import *
try:
  from protobuflang import *
except ImportError:
  pass

