import math

from raco.utility import emit, Printable

"""
An expression language for datalog: Function calls, arithmetic, simple string functions
"""

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
    #
    # TODO(andrew) am I missing something crazy here? self.postorder(doit) does
    # nothing.
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

class NaryOperator(Expression):
  pass

class UDF(NaryOperator):
  pass

class POW(BinaryOperator):
  literals = ['POW']
  def evaluate(self, _tuple, scheme):
    return pow(self.left.evaluate(_tuple, scheme),
               self.right.evaluate(_tuple, scheme))

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

class UnaryFunction(UnaryOperator):
  def __str__(self):
    return "%s(%s)" % (self.__class__.__name__, self.input)

class ABS(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return abs(self.input.evaluate(_tuple, scheme))

class CEIL(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.ceil(self.input.evaluate(_tuple, scheme))

class COS(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.cos(self.input.evaluate(_tuple, scheme))

class FLOOR(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.floor(self.input.evaluate(_tuple, scheme))

class LOG(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.log(self.input.evaluate(_tuple, scheme))

class SIN(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.sin(self.input.evaluate(_tuple, scheme))

class SQRT(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.sqrt(self.input.evaluate(_tuple, scheme))

class TAN(UnaryFunction):
  def evaluate(self, _tuple, scheme):
    return math.tan(self.input.evaluate(_tuple, scheme))

class AggregateExpression(Expression):
  def evaluate(self, _tuple, scheme):
    """Stub evaluate function for aggregate expressions.

    Aggregate functions do not evaluate individual tuples; rather they
    operate on collections of tuples in the evaluate_aggregate function.
    We return a dummy string so that all tuples containing this aggregate
    hash to the same value.
    """
    return self.opname()

  def evaluate_aggregate(self, tuple_iterator, scheme):
    """Evaluate an aggregate over a bag of tuples"""
    raise NotImplementedError()

class MAX(UnaryFunction,AggregateExpression):
  def evaluate_aggregate(self, tuple_iterator, scheme):
    inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
    return max(inputs)

class MIN(UnaryFunction,AggregateExpression):
  def evaluate_aggregate(self, tuple_iterator, scheme):
    inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
    return min(inputs)

class COUNTALL(ZeroaryOperator, AggregateExpression):
  def evaluate_aggregate(self, tuple_iterator, scheme):
    return len(tuple_iterator)

class COUNT(UnaryFunction,AggregateExpression):
  def evaluate_aggregate(self, tuple_iterator, scheme):
    inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
    count = 0
    for t in inputs:
      if t is not None:
        count += 1
    return count

class SUM(UnaryFunction,AggregateExpression):
  def evaluate_aggregate(self, tuple_iterator, scheme):
    inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)

    sum = 0
    for t in inputs:
      if t is not None:
        sum += t
    return sum

class AVERAGE(UnaryFunction,AggregateExpression):
  def evaluate_aggregate(self, tuple_iterator, scheme):
    inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
    filtered = (x for x in inputs if x is not None)

    sum = 0
    count = 0
    for t in filtered:
        sum += t
        count += 1
    return sum / count

class STDEV(UnaryFunction,AggregateExpression):
  def evaluate_aggregate(self, tuple_iterator, scheme):
    inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
    filtered = [x for x in inputs if x is not None]

    n = len(filtered)
    if (n < 2):
      return 0.0

    mean = float(sum(filtered)) / n

    std = 0.0
    for a in filtered:
      std = std + (a - mean)**2
    std = math.sqrt(std / n)
    return std

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
