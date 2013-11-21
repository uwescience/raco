"""
An expression language for Raco: functions, booleans, aggregates, etc.

Most non-trivial operators and functions are in separate files in this package.
"""

from raco.utility import Printable

class Expression(Printable):
  def typeof(self):
    # By default, we don't know the type
    return None

  def opstr(self):
    if not hasattr(self, "literals"):
      return self.opname()
    else:
      return self.literals[0]

  def evaluate(self, _tuple, scheme):
    '''Evaluate an expression in the context of a given tuple and schema.

    This is used for unit tests written against the fake database.
    '''
    raise NotImplementedError()

  def postorder(self, f):
    """Apply a function to each node in an expression tree.

    The function argument should return a scalar.  The return value
    of postorder is an iterator over these scalars.
    """
    yield f(self)

  def apply(self, f):
    """Replace children with the result of a function"""
    pass

  def add_offset(self, offset):
    """Add a constant offset to every positional reference in this tree"""
    def doit(self):
        if isinstance(self, UnnamedAttributeRef):
            self.position += offset
        return self

    # We have to manually walk the postorder because otherwise nothing ever
    # .. gets executed. Stupid generators.
    for x in self.postorder(doit):
        pass

class ZeroaryOperator(Expression):
  def __init__(self):
    pass
  def __eq__(self, other):
    return self.__class__ == other.__class__ 

  def __hash__(self):
    return hash(self.__class__)

  def __str__(self):
    return "%s" % (self.opstr(),)

  def __repr__(self):
    return self.__str__()

class UnaryOperator(Expression):
  def __init__(self, input):
    self.input = input

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.input == other.input

  def __hash__(self):
    return hash(self.__class__) + hash(self.input)

  def __str__(self):
    return "%s%s" % (self.opstr(), self.input)

  def __repr__(self):
    return self.__str__()

  def postorder(self, f):
    for x in self.input.postorder(f):
      yield x
    yield f(self)

  def apply(self, f):
    self.input = f(self.input)


class BinaryOperator(Expression):
  def __init__(self, left, right):
    self.left = left
    self.right = right

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.left == other.left and self.right == other.right

  def __hash__(self):
    return hash(self.__class__) + hash(self.left) + hash(self.right)

  def __str__(self):
    return "(%s %s %s)" % (self.left, self.opstr(), self.right)

  def __repr__(self):
    return self.__str__()

  def postorder(self, f):
    for x in self.left.postorder(f):
      yield x
    for x in self.right.postorder(f):
      yield x
    yield f(self)

  def apply(self, f):
    self.left = f(self.left)
    self.right = f(self.right)

  def leftoffset(self, offset):
    """Add a constant offset to all positional references in the left subtree"""
    # TODO this is a very weird mechanism. It's really: take all terms that
    # reference the left child and add the offset to them. The implementation
    # is awkward and, is it correct? Elephant x Rhino!
    if isinstance(self.left, BinaryOperator):
        self.left.leftoffset(offset)
        self.right.leftoffset(offset)
    else:
        self.left.add_offset(offset)

  def rightoffset(self, offset):
    """Add a constant offset to all positional references in the right subtree"""
    # TODO see leftoffset
    if isinstance(self.right, BinaryOperator):
        self.left.rightoffset(offset)
        self.right.rightoffset(offset)
    else:
        self.right.add_offset(offset)

class NaryOperator(Expression):
  pass

class Literal(ZeroaryOperator):
  def __init__(self, value):
    self.value = value

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.value == other.value

  def __hash__(self):
    return hash(self.__class__) + hash(self.value)

  def vars(self):
    return []

  def __repr__(self):
    return str(self.value)

  def __str__(self):
    return str(self.value)

  def typeof(self):
    # TODO: DANGEROUS
    return type(self.value)

  def evaluate(self, _tuple, scheme):
    return self.value

  def apply(self, f):
    pass

class StringLiteral(Literal):
  pass

class NumericLiteral(Literal):
  pass

class AttributeRef(Expression):
  def evaluate(self, _tuple, scheme):
    return _tuple[self.get_position(scheme)]

  def get_position(self, scheme):
    raise NotImplementedError()

class NamedAttributeRef(AttributeRef):
  def __init__(self, attributename):
    self.name = attributename

  def __repr__(self):
    return "%s" % (self.name)

  def __str__(self):
    return "%s" % (self.name)

  def get_position(self, scheme):
    return scheme.getPosition(self.name)

class UnnamedAttributeRef(AttributeRef):
  def __init__(self, position):
    self.position = position

  def __repr__(self):
    return "$%s" % (self.position)

  def __str__(self):
    return "$%s" % (self.position)

  def get_position(self, scheme):
    return self.position

class DottedAttributeRef(AttributeRef):
  def __init__(self, relation_name, field):
    """Initializse a DottedAttributeRef.

    relation_name is a string that refers to a relation.
    field refers to a column; its value is either a string or an integer.
    """
    self.relation_name = relation_name
    self.field = field

  def __repr__(self):
    return "%s.%s" % (self.relation_name, str(self.field))

  def __str__(self):
    return "%s.%s" % (self.relation_name, str(self.field))

  def evaluate(self, _tuple, scheme):
    """Panic on attempted evaluation.

    DottedAttributeRefs are compiled away into UnnamedAttributeRefs before
    evaluation.  See unpack_from.py.
    """
    raise NotImplementedError()

class UDF(NaryOperator):
  pass

class PLUS(BinaryOperator):
  literals = ["+"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) +
            self.right.evaluate(_tuple, scheme))

class MINUS(BinaryOperator):
  literals = ["-"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) -
            self.right.evaluate(_tuple, scheme))

class DIVIDE(BinaryOperator):
  literals = ["/"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) /
            self.right.evaluate(_tuple, scheme))


class TIMES(BinaryOperator):
  literals = ["*"]

  def evaluate(self, _tuple, scheme):
    return (self.left.evaluate(_tuple, scheme) *
            self.right.evaluate(_tuple, scheme))


class NEG(UnaryOperator):
  literals = ["-"]

  def evaluate(self, _tuple, scheme):
    return -1 * self.input.evaluate(_tuple, scheme)

class Unbox(ZeroaryOperator):
  def __init__(self, relational_expression, field):
    """Initialize an unbox expression.

    relational_expression is a Myrial AST that evaluates to a relation.

    field is an optional column name/index within the relation.  If None,
    the system uses index 0.
    """
    self.relational_expression = relational_expression
    self.field = field

  def evaluate(self, _tuple, scheme):
    """Raise an error on attempted evaluation.

    Unbox expressions are not "evaluated" in the usual sense.  Rather, they
    are replaced with raw attribute references at evaluation time.
    """
    raise NotImplementedError()
