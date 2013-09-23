import unittest
import sys
from raco import RACompiler
import raco.expression as e

def RATest(query):
  dlog = RACompiler()
  dlog.fromDatalog(query)
  dlog.logicalplan
  # TODO: Testing for string equality. Is this reasonable?
  return "%s" % dlog.logicalplan

class DatalogTest(unittest.TestCase):
  def test_join(self):
    join = """A(x,z) :- R(x,y), S(y,z)"""
    desiredresult = """[('A', Project($0,$3)[Join($1 = $0)[Scan(R), Scan(S)]])]"""
    testresult = RATest(join)
    self.assertEqual(testresult, desiredresult)

  def test_selfjoin(self):
    join = """A(x,z) :- R(x,y), R(y,z)"""
    desiredresult = """[('A', Project($0,$3)[Join($1 = $0)[Scan(R), Scan(R)]])]"""
    testresult = RATest(join)
    self.assertEqual(testresult, desiredresult)

  #def test_triangle(self):
  #  join = """A(x,y,z) :- R(x,y), S(y,z), T(z,x)"""
  #  desiredresult = """[('A', Project($0,$1,$3)[Select($0 = $5)[Join($3 = $0)[Join($1 = $0)[Scan(R), Scan(S)], Scan(T)]]])]"""
  #  testresult = RATest(join)
  #  self.assertEqual(testresult, desiredresult)

  def test_explicit_conditions(self):
    join = """A(x,y,z) :- R(x,y), S(w,z), x<y,y<z,y=w"""
    desiredresult = """[('A', Project($0,$1,$3)[Join($1 < $1 and $1 = $0)[Select($0 < $1)[Scan(R)], Scan(S)]])]"""
    testresult = RATest(join)
    self.assertEqual(testresult, desiredresult)

  def test_select(self):
    select = "A(x) :- R(x,3)"
    desiredresult = """[('A', Project($0)[Select($1 = 3)[Scan(R)]])]"""
    testresult = RATest(select)
    self.assertEqual(testresult, desiredresult)

  def test_select2(self):
    select = "A(x) :- R(x,y), S(y,z,4), z<3"
    desiredresult = """[('A', Project($0)[Join($1 = $0)[Scan(R), Select($2 = 4 and $1 < 3)[Scan(S)]]])]"""
    testresult = RATest(select)
    self.assertEqual(testresult, desiredresult)

  def test_recursion(self):
    select = """
    A(x) :- R(x,3)
    A(x) :- R(x,y), A(y)
    """
    desiredresult = """[('A', Project($0)[Join($1 = $0)[Scan(R), Select($2 = 4 and $1 < 3)[Scan(S)]]])]"""
    testresult = RATest(select)
    self.assertEqual(testresult, desiredresult)

class ExpressionTest(unittest.TestCase):
  def test_postorder(self):
    expr1 = e.MINUS(e.MAX(e.NamedAttributeRef("salary")), e.MIN(e.NamedAttributeRef("salary")))
    expr2 = e.PLUS(e.LOG(e.NamedAttributeRef("salary")), e.ABS(e.NamedAttributeRef("salary")))
    
    def isAggregate(expr):
      return isinstance(expr,e.AggregateExpression)

    def classname(expr):
      return expr.__class__.__name__

    e1cls = [x for x in expr1.postorder(classname)]

    e2cls = [x for x in expr2.postorder(classname)]

    e1any = any(expr1.postorder(isAggregate))

    e2any = any(expr2.postorder(isAggregate))

    self.assertEqual(str(e1cls), """['NamedAttributeRef', 'MAX', 'NamedAttributeRef', 'MIN', 'MINUS']""")
    self.assertEqual(str(e2cls), """['NamedAttributeRef', 'LOG', 'NamedAttributeRef', 'ABS', 'PLUS']""")
    self.assertEqual(e1any, True)
    self.assertEqual(e2any, False)

if __name__ == '__main__':
  unittest.main()

