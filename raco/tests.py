import unittest
from raco import RACompiler

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

  def test_triangle(self):
    join = """A(x,y,z) :- R(x,y), S(y,z), T(z,x)"""
    desiredresult = """[('A', Project($0,$1,$2)[Select($1 = $4)[Join($2 = $1)[Join($0 = $1)[Scan(R), Scan(T)], Scan(S)]]])]"""
    testresult = RATest(join)
    self.assertEqual(testresult, desiredresult)

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

if __name__ == '__main__':
   unittest.main()
