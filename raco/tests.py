import unittest
from raco import RACompiler
import raco.expression as e

def RATest(query):
  dlog = RACompiler()
  dlog.fromDatalog(query)
  # TODO: Testing for string equality, but we should do something like what Andrew does -- evaluate the expressions on test data.
  return "%s" % dlog.logicalplan

class DatalogTest(unittest.TestCase):
  def test_join(self):
    join = """A(x,z) :- R(x,y), S(y,z)"""
    desiredresult = """[('A', Project($0,$3)[Join(($1 = $2))[Scan(R), Scan(S)]])]"""
    testresult = RATest(join)
    self.assertEqual(testresult, desiredresult)

  def test_selfjoin(self):
    join = """A(x,z) :- R(x,y), R(y,z)"""
    desiredresult = """[('A', Project($0,$3)[Join(($1 = $2))[Scan(R), Scan(R)]])]"""
    testresult = RATest(join)
    self.assertEqual(testresult, desiredresult)

  def test_triangle(self):
    join = """A(x,y,z) :- R(x,y), S(y,z), T(z,x)"""
    desiredresult = """[('A', Project($0,$1,$3)[Select(($3 = $4))[Join(($0 = $5))[Join(($1 = $2))[Scan(R), Scan(S)], Scan(T)]]])]"""
    testresult = RATest(join)
    self.assertEqual(testresult, desiredresult)

  def test_explicit_conditions(self):
    join = """A(x,y,z) :- R(x,y), S(w,z), x<y,y<z,y=w"""
    desiredresult = """[('A', Project($0,$1,$3)[Join((($1 < $3) and ($1 = $2)))[Select(($0 < $1))[Scan(R)], Scan(S)]])]"""
    testresult = RATest(join)
    self.assertEqual(testresult, desiredresult)

  def test_select(self):
    select = "A(x) :- R(x,3)"
    desiredresult = """[('A', Project($0)[Select(($1 = 3))[Scan(R)]])]"""
    testresult = RATest(select)
    self.assertEqual(testresult, desiredresult)

  def test_select2(self):
    select = "A(x) :- R(x,y), S(y,z,4), z<3"
    desiredresult = """[('A', Project($0)[Join(($1 = $2))[Scan(R), Select((($2 = 4) and ($1 < 3)))[Scan(S)]]])]"""
    testresult = RATest(select)
    self.assertEqual(testresult, desiredresult)

  #def test_recursion(self):
  #  select = """
  #  A(x) :- R(x,3)
  #  A(x) :- R(x,y), A(y)
  #  """
  #  desiredresult = """[('A', Project($0)[Join($1 = $0)[Scan(R), Select($2 = 4 and $1 < 3)[Scan(S)]]])]"""
  #  testresult = RATest(select)
  #  self.assertEqual(testresult, desiredresult)

  def test_groupby_count(self):
    query = """
    InDegree(dst, count(src)) :- Edge(src,dst)
    """
    desiredresult = """[('InDegree', GroupBy($1; COUNT($0))[Scan(Edge)])]"""
    testresult = RATest(query)
    self.assertEqual(testresult, desiredresult)

  def test_groupby_sum(self):
    query = """
    TotalSalary(emp_id, sum(salary)) :- Employee(emp_id, dept_id,salary)
    """
    desiredresult = """[('TotalSalary', GroupBy($0; SUM($2))[Scan(Employee)])]"""
    testresult = RATest(query)
    self.assertEqual(testresult, desiredresult)

  def test_sum(self):
    query = """
    TotalSalary(sum(salary)) :- Employee(emp_id, dept_id,salary)
    """
    desiredresult = """[('TotalSalary', GroupBy(; SUM($2))[Scan(Employee)])]"""
    testresult = RATest(query)
    self.assertEqual(testresult, desiredresult)

  def test_union(self):
    query = """
A(x) :- B(x,y)
A(x) :- C(y,x)
"""
    desiredresult = """[('A', Union[Project($0)[Scan(B)], Project($1)[Scan(C)]])]"""
    testresult = RATest(query)
    self.assertEqual(testresult, desiredresult)

  def test_chained(self):
    query = """
JustXBill(x) :- TwitterK(x,y)
JustXBill2(x) :- JustXBill(x)
JustXBillSquared(x) :- JustXBill(x), JustXBill2(x)
"""
    desiredresult = """[('JustXBillSquared', Project($0)[Join(($0 = $1))[Apply(x=$0)[Project($0)[Scan(TwitterK)]], Apply(x=$0)[Project($0)[Apply(x=$0)[Project($0)[Scan(TwitterK)]]]]]])]"""
    testresult = RATest(query)
    self.assertEqual(testresult, desiredresult)

  def test_chained_rename(self):
    query = """
    A(x,z) :- R(x,y,z);
    B(w) :- A(3,w)
"""
    desiredresult = """[('B', Project($1)[Select(($0 = 3))[Apply(x=$0,w=$1)[Project($0,$2)[Scan(R)]]]])]"""
    testresult = RATest(query)
    self.assertEqual(testresult, desiredresult)

  def test_filter_expression(self):
    query = """
filtered(src, dst, time) :- nccdc(src, dst, proto, time, a, b, c), time > 1366475761, time < 1366475821
"""
    desiredresult="[('filtered', Project($0,$1,$3)[Select((($3 > 1366475761) and ($3 < 1366475821)))[Scan(nccdc)]])]"
    testresult = RATest(query)
    self.assertEquals(testresult, desiredresult)

  def test_aggregate_no_groups(self):
    query = "Total(count(y)) :- R(x,y)"
    desiredresult="[('Total', GroupBy(; COUNT($1))[Scan(R)])]"
    testresult = RATest(query)
    self.assertEquals(testresult, desiredresult)

  def test_multigroupby_count(self):
    query = "Total(y, z, count(x)) :- R(x,y,z)"
    desiredresult="[('Total', GroupBy($1,$2; COUNT($0))[Scan(R)])]"
    testresult = RATest(query)
    self.assertEquals(testresult, desiredresult)

  def test_multigroupby_sum_reorder(self):
    query = """Total(sum(x), z, y) :- R(x,y,z);
Output(s) :- Total(s,z,y)
    """
    desiredresult="[('Output', Project($0)[Apply(s=$0,z=$1,y=$2)[GroupBy($2,$1; SUM($0))[Scan(R)]]])]"
    testresult = RATest(query)
    self.assertEquals(testresult, desiredresult)

class ExpressionTest(unittest.TestCase):
  def test_postorder(self):
    expr1 = e.MINUS(e.MAX(e.NamedAttributeRef("salary")), e.MIN(e.NamedAttributeRef("salary")))
    expr2 = e.PLUS(e.LOG(e.NamedAttributeRef("salary")), e.ABS(e.NamedAttributeRef("salary")))
    
    def isAggregate(expr):
      return isinstance(expr, e.AggregateExpression)

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

