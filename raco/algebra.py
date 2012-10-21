import boolean
import scheme
from utility import emit, Printable
from rules import Rule

"""
Generate variables names
"""
i = 0
def gensym():
  global i
  i += 1
  return "V%s" % i

class Operator(Printable):
  """Operator base classs"""
  def __init__(self):
    self.bound = None
    # Extra code to emit to cleanup
    self.cleanup = ""
    self.alias = self

  def copy(self, other):
    self.bound = None

  def set_alias(self, alias):
    """Set a user-defined identififer for this operator.  Used in optimization and transformation of plans"""
    self.alias = alias

class ZeroaryOperator(Operator):
  """Operator with no arguments"""
  def __init__(self):
    Operator.__init__(self)

  def compile(self, resultsym):
    if self.bound:
      code = self.language.assignment(resultsym, self.bound)     
    else:
      code = "%s" % (self.compileme(resultsym),)
      self.bound = resultsym
    return code

  def apply(self, f):
    """Apply a function to your children"""
    return self

  def __str__(self):
    return "%s" % self.opname()

  def copy(self, other):
    """Deep copy"""
    Operator.copy(self, other)

  def postorder(self, f):
    """Postorder traversal, applying a function to each operator.  The function returns an iterator"""
    for x in f(self): yield x

  def preorder(self, f):
    """Preorder traversal, applying a function to each operator"""
    self.postorder(f)

  def collectParents(self, parentmap):
    pass

class UnaryOperator(Operator):
  """Operator with one argument"""
  def __init__(self, input):
    self.input = input
    Operator.__init__(self)

  def compile(self, resultsym):
    """Compile this operator to the language specified."""
    #TODO: Why is the language not an argument? 
    if self.bound:
      code = self.language.assignment(resultsym, self.bound)
    else:
      inputsym = gensym()
      # compile the previous operator
      prev = self.input.compile(inputsym)
      # compile me
      me = self.compileme(resultsym, inputsym)
      code = emit(prev, me)
    return code

  def apply(self, f):
    """Apply a function to your children"""
    self.input = f(self.input)
    return self

  def __str__(self):
    return "%s(%s)" % (self.opname(), self.input)

  def copy(self, other):
    """deep copy"""
    self.input = other.input
    Operator.copy(self, other)

  def postorder(self, f):
    """Postorder traversal. Apply a function to your children. Function returns an iterator."""
    for x in self.input.postorder(f): yield x
    for x in f(self): yield x

  def preorder(self, f):
    """Preorder traversal. Apply a function to your children. Function returns an iterator."""
    for x in f(self): yield x
    for x in self.input.postorder(f): yield x

  def collectParents(self, parentmap):
    """Construct a dict mapping children to parents. Used in optimization"""
    parentmap.setdefault(self.input, []).append(self)
    self.input.collectParents(parentmap)


class BinaryOperator(Operator):
  """Operator with two arguments"""
  def __init__(self, left, right):
    self.left = left
    self.right = right
    Operator.__init__(self)

  def compile(self, resultsym):
    """Compile this plan.  Result sym is the variable name to use to hold the result of this operator."""
    #TODO: Why is language not an argument?
    if self.bound:
      code = self.language.assignment(resultsym, self.bound)
    else:
      leftsym = gensym()
      rightsym = gensym()
      code = """
%s 
%s
%s
""" % (self.left.compile(leftsym)
      , self.right.compile(rightsym)
      , self.compileme(resultsym, leftsym, rightsym))
    return code

  def apply(self, f):
    """Apply a function to your children"""
    self.left = f(self.left)
    self.right = f(self.right)
    return self

  def __str__(self):
    return "%s(%s,%s)" % (self.opname(), self.left, self.right)

  def copy(self, other):
    """deep copy"""
    self.left = other.left
    self.right = other.right
    Operator.copy(self, other)

  def postorder(self, f):
    """postorder traversal.  Apply a function to each operator.  Function returns an iterator."""
    for x in self.left.postorder(f): yield x
    for x in self.right.postorder(f): yield x
    for x in f(self): yield x

  def preorder(self, f):
    """preorder traversal.  Apply a function to each operator.  Function returns an iterator."""
    for x in f(self): yield x
    for x in self.left.postorder(f): yield x
    for x in self.right.postorder(f): yield x

  def collectParents(self, parentmap):
    """Construct a dict mapping children to parents. Used in optimization."""
    parentmap.setdefault(self.left, []).append(self)
    parentmap.setdefault(self.right, []).append(self)
    self.left.collectParents(parentmap)
    self.right.collectParents(parentmap)

"""Logical Relational Algebra"""

class Union(BinaryOperator):
  def scheme(self): 
    assert left.scheme() == right.scheme()
    return left.scheme()

class Join(BinaryOperator):
  """Logical Join operator"""
  def __init__(self, condition=None, left=None, right=None):
    self.condition = condition
    BinaryOperator.__init__(self, left, right)

  def __str__(self):
    return "%s(%s)[%s, %s]" % (self.opname(), self.condition, self.left, self.right)

  def copy(self, other):
    """deep copy"""
    self.condition = other.condition
    BinaryOperator.copy(self, other)

  def scheme(self):
    """Return the scheme of the result."""
    return self.left.scheme() + self.right.scheme()

class Select(UnaryOperator):
  """Logical selection operator"""
  def __init__(self, condition=None, input=None):
    self.condition = condition
    UnaryOperator.__init__(self, input)

  def __str__(self):
    if isinstance(self.condition,dict): 
      cond = self.condition["condition"]
    else:
      cond = self.condition
    return "%s(%s)[%s]" % (self.opname(), cond, self.input)

  def copy(self, other):
    """deep copy"""
    self.condition = other.condition
    UnaryOperator.copy(self, other)

  def scheme(self):
    """scheme of the result."""
    return self.input.scheme()

class CrossProduct(BinaryOperator):
  """Logical Cross Product operator"""
  def __init__(self, left=None, right=None):
    BinaryOperator.__init__(self, left, right)

  def copy(self, other):
    """deep copy"""
    BinaryOperator.copy(self, other)

  def __str__(self):
    return "%s[%s, %s]" % (self.opname(), self.left, self.right)

  def scheme(self):
    """Return the scheme of the result."""
    return self.left.scheme() + self.right.scheme()

class Project(UnaryOperator):
  """Logical projection operator"""
  def __init__(self, columnlist=None, input=None):
    self.columnlist = columnlist
    UnaryOperator.__init__(self, input)

  def __str__(self):
    colstring = ",".join([str(x) for x in self.columnlist])
    return "%s(%s)[%s]" % (self.opname(), colstring, self.input)

  def copy(self, other):
    """deep copy"""
    self.columnlist = other.columnlist
    UnaryOperator.copy(self, other)

  def scheme(self):
    """scheme of the result. Raises a TypeError if a name in the project list is not in the source schema"""
    return scheme.Scheme([attref.resolve(self.input.scheme()) for attref in self.columnlist])

class GroupBy(UnaryOperator):
  """Logical projection operator"""
  def __init__(self, groupinglist=None, expressionlist=None, input=None):
    self.groupinglist = groupinglist
    self.expressionlist = expressionlist
    UnaryOperator.__init__(self, input)

  def __str__(self):
    colstring = ",".join([str(x) for x in self.columnlist])
    return "%s(%s)[%s]" % (self.opname(), colstring, self.input)

  def copy(self, other):
    """deep copy"""
    self.groupinglist = other.groupinglist
    self.expressionlist = other.expressionlist
    UnaryOperator.copy(self, other)

  @classmethod
  def typeof(self):
    """Infer the type of an aggregate expression"""
    return float

  def scheme(self):
    """scheme of the result. Raises a TypeError if a name in the project list is not in the source schema"""
    groupingscheme = [attref.resolve(self.input.scheme()) for attref in self.groupinglist]
    expressionscheme = [("expr%s" % i, GroupBy.typeof(expr)) for i,expr in enumerate(self.expressionlist)]

class EmptyRelation(ZeroaryOperator):
  """Empty Relation.  Used in certain optimizations."""
  def __str__(self):
    return "EmptySet"

  def copy(self, other):
    """deep copy"""
    pass

  def scheme(self):
    """scheme of the result."""
    return scheme.Scheme()

class Scan(ZeroaryOperator):
  """Logical Scan operator"""
  def __init__(self, relation=None):
    self.relation = relation
    ZeroaryOperator.__init__(self)

  def __str__(self):
    return "%s(%s)" % (self.opname(), self.relation.name)

  def __repr__(self):
    return str(self)

  def copy(self, other):
    """deep copy"""
    self.relation = other.relation
    ZeroaryOperator.copy(self, other)

  def scheme(self):
    """Scheme of the result, which is just the scheme of the relation."""
    return self.relation.scheme
 

class CollapseSelect(Rule):
  """A rewrite rule for combining two selections"""
  def fire(self, expr):
    if isinstance(expr, Select):
      if isinstance(expr.input, Select):
         newcondition = boolean.AND(expr.condition, expr.input.condition)
         return Select(newcondition, expr.input.input)
    return expr

  def __str__(self):
    return "Select, Select => Select"

def attribute_references(condition):
  """Generates a list of attributes referenced in the condition"""
  if isinstance(condition, BinaryBooleanOperator):
    for a in attribute_references(condition.left): yield a
    for a in attribute_references(condition.right): yield a
  elif isinstance(condition, Attribute):
    yield condition.name
"""
#def coveredby(

class PushSelect(Rule):
  def fire(self, expr):
    if isinstance(expr, Select):
      if isinstance(expr.input, Join):
        join = expr.input
        select = expr
        if join.left.scheme().contains(attributes):
          # push left
        if join.right.scheme().contains(attributes):
          # push right
"""     

class LogicalAlgebra:
  operators = [
  Join,
  Select,
  Scan
]
  rules = [
  CollapseSelect()
]


