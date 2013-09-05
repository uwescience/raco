import raco.expression as e
from raco.language import PythonAlgebra
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


def test():
  sch = Scheme([("y",float), ("x",int)])
  R = raco.catalog.Relation("R", sch)
  J = Join(EQ(e.UnnamedAttributeRef(0), e.NamedAttributeRef("x")), Scan(R), Scan(R))
  exprs = optimize([('A',J)], target=PythonAlgebra, source=LogicalAlgebra)
  print compile(exprs)

test()
