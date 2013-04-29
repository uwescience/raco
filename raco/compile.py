import algebra
from utility import emit

"""
Apply rules to an expression
If successful, output will 
only involve operators in the
target algebra
"""

def optimize_by_rules(expr, rules):
  for rule in rules:
    def recursiverule(expr):
      newexpr = rule(expr)
      newexpr.apply(recursiverule)
      return newexpr
    expr = recursiverule(expr)
  return expr

def optimize_by_rules_breadth_first(expr, rules):
  def optimizeto(expr):
     return optimize_by_rules(expr, rules)

  for rule in rules:
    newexpr = rule(expr)
    expr = newexpr
  expr = expr.apply(optimizeto)
  return expr


def optimize(exprs, target, source, eliminate_common_subexpressions=False):
  """Fire the rule-based optimizer on a list of exprs.  Fire all rules in the source algebra 
(logical) and the target algebra (physical)"""
  def opt(expr):
    so = optimize_by_rules(expr, source.rules)
    newexpr = optimize_by_rules(so, target.rules)
    if eliminate_common_subexpressions:
      newexpr = common_subexpression_elimination(newexpr)
    return newexpr
  return [(var, opt(exp)) for var, exp in exprs]

def compile(exprs):
  """Compile a list of RA expressions.  Each expression is a pair Var, Plan
Emit any initialization and call compile method on top-level operator
"""
  algebra.reset()
  exprcode = []
  for resultsym, expr in exprs:
    init = expr.language.initialize(resultsym)
    body = expr.compile(resultsym)
    final = expr.language.finalize(resultsym)
    exprcode.append(emit(init,body,final))
  return  emit(*exprcode)

def search(expr, tofind):
  """yield a sequence of subexpressions equal to tofind"""
  def match(node):
    if node == tofind: yield node
  for x in expr.preorder(match):
    yield x

def common_subexpression_elimination(expr):
  """remove redundant subexpressions"""
  def id(expr):
    yield expr
  eqclasses = []
  allfound = []
  for x in expr.preorder(id):
    if not x in allfound:
      found = [x for x in search(expr, x)]
      eqclasses.append((x,found))
      allfound += found 

  def replace(expr):
    for witness, ec in eqclasses:
      if expr in ec:
        expr.apply(replace)
        # record the fact that we eliminated the redundant branches
        if witness != expr:
          #witness.trace("replaces", expr)
          for k,v in expr.gettrace():
            witness.trace(k, v) 
        return witness

  return expr.apply(replace)

def showids(expr):
  """Traverse the plan and show the operator ids"""
  def getid(node):
    yield node, id(node)
 
  return expr.preorder(getid)
  
