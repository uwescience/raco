from utility import Printable

"""
Boolean logic
"""

class BooleanExpression(Printable):
  pass

'''
# not supported, and not necessary
class BooleanLiteral(BooleanExpression):
  pass

class BTrue(BooleanLiteral):
  literals = ["True", "true", "TRUE"]

class BFalse(BooleanLiteral):
  literals = ["False", "false", "FALSE"]
'''

class UnaryBooleanOperator(BooleanExpression):
  def __init__(self, input):
    self.input = input

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.input == other.input

  def __str__(self):
    if not hasattr(self, "literals"):
      opstr = self.opname()
    else:
     opstr = self.literals[0]
    return "%s%s" % (opstr, self.input)

  def __repr__(self):
    return self.__str__()

  def leftoffset(self, offset):
    """Add an offset to any positional references on the left-hand side.  Useful when constructing left-deep plans."""
    self.input.leftoffset(offset)

  def rightoffset(self, offset):
    """Add an offset to any positional references on the left-hand side.  Useful when constructing right-deep plans."""
    self.input.rightoffset(offset)

  def flip(self):
    """flip the order of comparison operators.  Used in optimizing join trees"""
    self.input.flip()


class BinaryBooleanOperator(BooleanExpression):
  def __init__(self, left, right):
    self.left = left
    self.right = right

  def vars(self):
    """Return a list of variables referenced in this expression. """
    # TODO: This is tangling the datalog parsing with the boolean expression model.  
    # Maybe subclass moel.Term and wrap the boolean expression.
    return self.left.vars() + self.right.vars()

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.left == other.left and self.right == other.right

  def __str__(self):
    if not hasattr(self, "literals"):
      opstr = self.opname()
    else:
     opstr = self.literals[0]
    return "%s %s %s" % (self.left, opstr, self.right)

  def __repr__(self):
    return self.__str__()

  def leftoffset(self, offset):
    """Add an offset to any positional references on the left-hand side.  Useful when constructing left-deep plans."""
    self.left.leftoffset(offset)
    self.right.leftoffset(offset)

  def rightoffset(self, offset):
    """Add an offset to any positional references on the left-hand side.  Useful when constructing right-deep plans."""
    self.left.rightoffset(offset)
    self.right.rightoffset(offset)

  def flip(self):
    """Return a new copy of this condition with the order of comparison operators flipped.  Used in optimizing join trees, when the left and right relations are swapped."""
    return self.__class__(self.left.flip(), self.right.flip())

class BinaryComparisonOperator(BinaryBooleanOperator):
  def flip(self):
    """Return a new condition that reverses the direction of this condition. 
E.g., 3>X becomes X<3. Useful for normalizing plans."""
    return reverse[self.__class__](self.right, self.left)

  def leftoffset(self, offset):
    """Add an offset to any positional references on the left-hand side.  Useful when constructing left-deep plans."""
    self.left.leftoffset(offset)
    
  def rightoffset(self, offset):
    """Add an offset to any positional references on the left-hand side.  Useful when constructing right-deep plans."""
    self.right.rightoffset(offset)

class NOT(UnaryBooleanOperator):
  literals = ["not", "NOT", "-"]

class AND(BinaryBooleanOperator):
  literals = ["and", "AND"]

class OR(BinaryBooleanOperator):
  literals = ["or", "OR"]

class EQ(BinaryComparisonOperator):
  literals = ["=", "=="]

class LT(BinaryComparisonOperator):
  literals = ["<", "lt"]

class GT(BinaryComparisonOperator):
  literals = [">", "gt"]

class GTEQ(BinaryComparisonOperator):
  literals = [">=", "gteq", "gte"]

class LTEQ(BinaryComparisonOperator):
  literals = ["<=", "lteq", "lte"]

class NEQ(BinaryComparisonOperator):
  literals = ["!=", "neq", "ne"]

reverse = {
  NEQ:NEQ,
  EQ:EQ,
  GTEQ:LTEQ,
  LTEQ:GTEQ,
  GT:LT,
  LT:GT
}

complement = {
  NEQ:EQ,
  EQ:NEQ,
  GTEQ:LT,
  LTEQ:GT,
  GT:LTEQ,
  LT:GTEQ
}

class Literal:
  def __init__(self, value):
    self.value = value

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.value == other.value 

  def vars(self):
    return []

  def __repr__(self):
    return str(self.value)

  def leftoffset(self, offset):
    """Add an offset to any positional references on the left-hand side.  Useful when constructing left-deep plans."""
    pass

  def rightoffset(self, offset):
    """Add an offset to this positional reference.  Used when building a plan from a set of joins"""
    pass 

class StringLiteral(Literal):
  pass

class NumericLiteral(Literal):
  pass

tautology = EQ(NumericLiteral(1),NumericLiteral(1))

def isTaut(ref) :
    try :
        return ref.literals == ['=','=='] and ref.left.value == 1 and ref.right.value == 1
    except AttributeError:
        return False

def binary_ops():
  """Return a list of all binary operator classes, like AND, OR"""
  import raco.boolean as boolean
  allclasses = [c for c in boolean.__dict__.values() if not hasattr(c, "__class__")]
  binopclasses = [opclass for opclass in allclasses
                   if issubclass(opclass,BinaryBooleanOperator)
                   and opclass is not BinaryBooleanOperator
                   and opclass is not BinaryComparisonOperator]

  return binopclasses
