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
    desiredresult = """[('A', Project(col0,col3)[Join(col1 = col0)[Scan(R), Scan(S)]])]"""
    testresult = RATest(join)
    self.assertEqual(testresult, desiredresult)

  def test_selfjoin(self):
    join = """A(x,z) :- R(x,y), R(y,z)"""
    desiredresult = """[('A', Project(col0,col3)[Join(col1 = col0)[Scan(R), Scan(R)]])]"""
    testresult = RATest(join)
    self.assertEqual(testresult, desiredresult)

  def test_triangle(self):
    join = """A(x,y,z) :- R(x,y), S(y,z), T(z,x)"""
    desiredresult = """[('A', Project(col0,col1,col2)[Select(col1 = col4)[Join(col2 = col1)[Join(col0 = col1)[Scan(R), Scan(T)], Scan(S)]]])]"""
    testresult = RATest(join)
    self.assertEqual(testresult, desiredresult)

  def test_explicit_conditions(self):
    join = """A(x,y,z) :- R(x,y), S(w,z), x<y,y<z,y=w"""
    desiredresult = """[('A', Project(col0,col1,col3)[Join(col1 < col1 and col1 = col0)[Select(col0 < col1)[Scan(R)], Scan(S)]])]"""
    testresult = RATest(join)
    self.assertEqual(testresult, desiredresult)

if __name__ == '__main__':
   unittest.main()
