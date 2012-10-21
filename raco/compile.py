
from utility import emit

"""
Apply rules to an expression
If successful, output will 
only involve operators in the
target algebra
"""
def optimize_by_rules(expr, rules):
  def optimizeto(expr):
     return optimize_by_rules(expr, rules)

  for rule in rules:
    newexpr = rule(expr)
    expr = newexpr
  expr = expr.apply(optimizeto)
  return expr


def optimize(expr, target, source):
  expr = optimize_by_rules(expr, source.rules)
  expr = optimize_by_rules(expr, target.rules)
  return expr

"""
Top-level compile function.
Emit any initialization and call 
compile method on top-level operator
"""
def compile(expr):
  result = "resultRelation"
  init = expr.language.initialize(result)
  code = expr.compile(result)
  final = expr.language.finalize(result)
  return emit(init,code,final)



