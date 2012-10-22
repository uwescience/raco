import boolean

class Rule:
  """
Argument is an expression tree
Returns a possibly modified expression tree
"""
  def __call__(self, expr):
    return self.fire(expr)

import algebra

class CrossProduct2Join(Rule):
  """A rewrite rule for removing Cross Product"""
  def fire(self, expr):
    if isinstance(expr, algebra.CrossProduct):
      return algebra.Join(boolean.EQ(boolean.NumericLiteral(1),boolean.NumericLiteral(1)), expr.left, expr.right)
    return expr

  def __str__(self):
    return "CrossProduct(left, right) => Join(1=1, left, right)"


class removeProject(Rule):
  """A rewrite rule for removing Projections"""
  def fire(self, expr):
    if isinstance(expr, algebra.Project):
      return expr.input
    return expr

  def __str__(self):
    return "Project => ()"

class OneToOne(Rule):
  def __init__(self, opfrom, opto):
    self.opfrom = opfrom
    self.opto = opto

  def fire(self, expr):
    if isinstance(expr, self.opfrom):
      newop = self.opto()
      newop.copy(expr)
      return newop
    return expr

  def __str__(self):
    return "%s => %s" % (self.opfrom.__name__,self.opto.__name__)

