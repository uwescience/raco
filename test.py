from raco.language import PythonAlgebra, PseudoCodeAlgebra, CCAlgebra
from raco.algebra import Select, Scan, Join, LogicalAlgebra
from raco.compile import compile, optimize
from raco.expression import EQ, AND, OR, NamedAttributeRef, StringLiteral, NumericLiteral

from sampledb import btc_schema, Rr

def pythonmain():
  testexpr1 = Join([("object","subject")], Scan("R"), Scan("R"))

  R = Scan(Rr)
  #sR = Select("lambda t: t.predicate == 'knows'", R)
  #sS = Select("lambda t: t.predicate == 'holdsAccount'", R)
  #sT = Select("lambda t: t.predicate == 'accountServiceHomepage'", R)
  sR = Select(EQ(NamedAttributeRef("predicate"), StringLiteral("knows")), R)
  sS = Select(EQ(NamedAttributeRef("predicate"), StringLiteral("holdsAccount")), R)
  sT = Select(EQ(NamedAttributeRef("predicate"), StringLiteral("accountServiceHomepage")), R)

  sRsS = Join([("object","subject")], sR, sS)

  sRsSsT = Join([("object1", "subject")], sRsS, sT)

  ssR = Select(EQ(NamedAttributeRef("predicate"), StringLiteral("holdsAccount")), sR)

  test = Select(OR(EQ(NamedAttributeRef("predicate"), StringLiteral("knows")),EQ(NamedAttributeRef("predicate"), StringLiteral("holdsAccount"))), R)

  #print ssR
  ossR = optimize(test, target=PythonAlgebra, source=LogicalAlgebra)
  #ossR = optimize(ssR, target=PseudoCodeAlgebra, source=LogicalAlgebra)
  #print ossR
  #print compile(ossR)
  #print compile(sRsSsT)
  #print [gensym() for i in range(5)]

  #sR = Select(EQ(NamedAttributeRef("predicate"), NumericLiteral(330337405)), R)
  osR = optimize(sRsSsT, target=PythonAlgebra, source=LogicalAlgebra)
  print compile(osR)

def main():
 # c implementation doesn't support string literals
 R = Scan(btc_schema["trial"])
 sR = Select(EQ(NamedAttributeRef("predicate"), NumericLiteral(1133564893)), R)
 sS = Select(EQ(NamedAttributeRef("predicate"), NumericLiteral(77645021)), R)
 sT = Select(EQ(NamedAttributeRef("predicate"), NumericLiteral(77645021)), R)


 sRsS = Join([("object","subject")], sR, sS)
 sRsSsT = Join([("object","subject")], sRsS, sT)
 """

 """
 result = optimize([('ans', sRsSsT)], target=CCAlgebra, source=LogicalAlgebra)
 #result = optimize(sR, target=CCAlgebra, source=LogicalAlgebra)
 return compile(result)
 def f(op):
   yield op #.__class__.__name__

 def g(op, vars):
   if op in vars: del vars[op]
   yield (op,vars.copy())

 def show(op):
   yield op.test

 #vars = dict([(x,1) for x in sR.postorder(f)])
 #vars.pop(sR, None)

 parents = {sRsSsT:[None]}
 sRsSsT.collectParents(parents)

 #def f(op):
 #  ps = parents[op]
 #  if ps:
 #    def freemem(resultsym
 #    ps[-1].cleanup += FREEOP % (op. __attrs__.setdefault("cleanup", [])
 #      if hasattr(parents[op]
 #    parents[op].cleanup
 #  yield (op,parents[op][-1]) #.__class__.__name__

 #print [x for x in sR.postorder(f)]
 #print parents

if __name__ == '__main__':
  #print testpython()
  print main()
