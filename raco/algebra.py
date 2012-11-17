import boolean
import scheme
from utility import emit, Printable
from rules import Rule

"""
Generate variables names
"""
i = 0
def reset():
  global i
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
    self._trace = []

  def __eq__(self, other):
    return self.__class__ == other.__class__

  def copy(self, other):
    self._trace = [pair for pair in other.gettrace()]
    self.bound = None

  def trace(self, key, val):
    self._trace.append((key, val))

  def gettrace(self):
    """Return a list of trace messages"""
    return self._trace

  def compiletrace(self):
    """Return the trace as a list of strings"""
    return "".join([self.language.comment("%s=%s" % (k,v)) for k,v in self.gettrace()])

  def set_alias(self, alias):
    """Set a user-defined identififer for this operator.  Used in optimization and transformation of plans"""
    self.alias = alias

class ZeroaryOperator(Operator):
  """Operator with no arguments"""
  def __init__(self):
    Operator.__init__(self)

  def __eq__(self, other):
    return self.__class__ == other.__class__

  def compile(self, resultsym):
    code = self.language.comment("Compiled subplan for %s" % self)
    code += self.language.log("Evaluating subplan %s" % self)
    self.trace("symbol", resultsym)
    if self.bound:
      code += self.language.new_relation_assignment(resultsym, self.bound)     
    else:
      code += "%s" % (self.compileme(resultsym),)
      self.bound = resultsym
      code += self.compiletrace()
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
    return self.postorder(f)

  def collectParents(self, parentmap):
    pass

class UnaryOperator(Operator):
  """Operator with one argument"""
  def __init__(self, input):
    self.input = input
    Operator.__init__(self)

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.input == other.input
  

  def compile(self, resultsym):
    """Compile this operator to the language specified."""
    #TODO: Why is the language not an argument? 
    code = self.language.comment("Compiled subplan for %s" % self)
    if self.bound:
      code += self.language.assignment(resultsym, self.bound)
    else:
      inputsym = gensym()
      # compile the previous operator
      prev = self.input.compile(inputsym)
      # compile me
      me = self.compileme(resultsym, inputsym)
      code += emit(prev, me)
    code += self.language.log("Evaluating subplan %s" % self)
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

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.left == other.left and self.right == other.right

  def compile(self, resultsym):
    """Compile this plan.  Result sym is the variable name to use to hold the result of this operator."""
    code = self.language.comment("Compiled subplan for %s" % self)
    code += self.language.log("Evaluating subplan %s" % self)
    #TODO: Why is language not an argument?
    if self.bound:
      code += self.language.assignment(resultsym, self.bound)
    else:
      leftsym = gensym()
      rightsym = gensym()
      code += """
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
    for x in self.left.preorder(f): yield x
    for x in self.right.preorder(f): yield x

  def collectParents(self, parentmap):
    """Construct a dict mapping children to parents. Used in optimization."""
    parentmap.setdefault(self.left, []).append(self)
    parentmap.setdefault(self.right, []).append(self)
    self.left.collectParents(parentmap)
    self.right.collectParents(parentmap)

class NaryOperator(Operator):
  """Operator with N arguments.  e.g., multi-way joins in one step."""
  def __init__(self, args):
    self.args = args
    Operator.__init__(self)

  def compile(self, resultsym):
    """Compile this plan.  Result sym is the variable name to use to hold the result of this operator."""
    #TODO: Why is language not an argument?
    code = self.language.comment("Compiled subplan for %s" % self)
    code += self.language.log("Evaluating subplan %s" % self)
    if self.bound:
      code += self.language.assignment(resultsym, self.bound)
    else:
      argsyms = [gensym() for arg in self.args]
      code += """
%s
%s
""" % ("\n".join([arg.compile(sym) for arg,sym in zip(self.args,argsyms)])
      , self.compileme(resultsym, argsyms))
    return code


  def copy(self, other):
    """deep copy"""
    self.args = [a for a in other.args]
    Operator.copy(self, other)

  def postorder(self, f):
    """postorder traversal.  Apply a function to each operator.  Function returns an iterator."""
    for arg in self.args:
      for x in arg.postorder(f): yield x
    for x in f(self): yield x

  def preorder(self, f):
    """preorder traversal.  Apply a function to each operator.  Function returns an iterator."""
    for x in f(self): yield x
    for arg in self.args:
      for x in arg.preorder(f): yield x

  def collectParents(self, parentmap):
    """Construct a dict mapping children to parents. Used in optimization."""
    for arg in self.args:
      parentmap.setdefault(arg, []).append(self)
      arg.collectParents(parentmap)

  def apply(self, f):
    """Apply a function to your children"""
    self.args = [f(arg) for arg in self.args]
    return self

  def __str__(self):
    return "%s[%s]" % (self.opname(), ",".join(["%s" % arg for arg in self.args]))

class NaryJoin(NaryOperator):
  def scheme(self):
    sch = scheme.Scheme()
    for arg in self.args:
      sch = sch + arg.scheme()
    return sch


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

  def __eq__(self, other):
    return BinaryOperator.__eq__(self,other) and self.condition == other.condition

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

  def __eq__(self, other):
    return UnaryOperator.__eq__(self,other) and self.condition == other.condition

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

  def __eq__(self, other):
    return UnaryOperator.__eq__(self,other) and self.columnlist == other.columnlist 

  def __str__(self):
    colstring = ",".join([str(x) for x in self.columnlist])
    return "%s(%s)[%s]" % (self.opname(), colstring, self.input)

  def __repr__(self):
    return "%s" % self

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

class Fixpoint(BinaryOperator):
  
  def __str__(self):
    return """Fixpoint[%s, %s]""" % (self.left, self.right)

class State(ZeroaryOperator):
  """A placeholder operator for recursive plan"""
  pass

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

  def __eq__(self,other):
    return ZeroaryOperator.__eq__(self,other) and self.relation == other.relation

  def __str__(self):
    return "%s(%s)" % (self.opname(), self.relation.name)

  def __repr__(self):
    return str(self)

  def copy(self, other):
    """deep copy"""
    self.relation = other.relation
    # TODO: need a cleaner and more general way of tracing information through 
    # the compilation process for debugging purposes
    if hasattr(other, "originalterm"): 
      self.originalterm = other.originalterm
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


