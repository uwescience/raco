import raco.expression as e
from raco import RACompiler
from raco.language import MyriaAlgebra
from raco.scheme import Scheme
from raco.algebra import Select, Scan, Join, Apply, LogicalAlgebra
from raco.boolean import EQ, AND, OR, StringLiteral, NumericLiteral
from sampledb import btc_schema, Rr
from raco.compile import compile, optimize
import raco.catalog

plus = e.PLUS(e.UnnamedAttributeRef(0), e.NamedAttributeRef("x"))
print plus

minus = e.MINUS(e.Literal(2), e.Literal(5))
print minus

absf = e.ABS(e.UnnamedAttributeRef(2))
print absf

mix = e.DIVIDE(e.TIMES(plus, absf), minus)
print mix


def testRA():
  sch = Scheme([("y",float), ("x",int)])
  R = raco.catalog.Relation("R", sch)
  J = Join(EQ(e.UnnamedAttributeRef(0), e.NamedAttributeRef("x")), Scan(R), Scan(R))
  A = Apply(J, z=e.PLUS(e.NamedAttributeRef("x"), e.NamedAttributeRef("y")), w=e.UnnamedAttributeRef(3))
  exprs = optimize([('A',A)], target=MyriaAlgebra(), source=LogicalAlgebra)
  print exprs
  print compile(exprs)

def testDatalog():
  query = "A(x) :- R(x,y),S(y,z)"

  print "/*\n%s\n*/" % str(query)

  # Create a compiler object
  dlog = RACompiler()

  # parse the query
  dlog.fromDatalog(query)
  print "************ LOGICAL PLAN *************"
  print dlog.logicalplan
  print

  # Optimize the query, includes producing a physical plan
  print "************ PHYSICAL PLAN *************"
  dlog.optimize(target=MyriaAlgebra(), eliminate_common_subexpressions=False)
  print dlog.physicalplan
  print

  # generate code in the target language
  code = dlog.compile()
  print "************ CODE *************"
  print code
  print


testRA()
