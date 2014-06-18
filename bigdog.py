from raco.datalog.grammar import parse
from raco.scheme import Scheme
from raco.catalog import ASCIIFile
from raco.language import FederatedAlgebra
from raco.algebra import LogicalAlgebra, Join, Apply, Scan, ExecScan
from raco.expression import boolean
from raco.expression import expression as e
from raco.compile import compile, optimize, common_subexpression_elimination, showids
from raco.types import FLOAT_TYPE, INT_TYPE
from raco import relation_key
import raco

from scidbpy import connect, SciDBQueryError, SciDBArray

sch = Scheme([("y",FLOAT_TYPE), ("x",INT_TYPE)])

# V0.0: Pass through queries to SciDB

# Construct a conventional RA plan
R = relation_key.RelationKey.from_string("public:adhoc:X")

R1 = Scan(R, sch)
R2 = Scan(R, sch)

J = Join(boolean.EQ(e.UnnamedAttributeRef(0), e.UnnamedAttributeRef(3)), R1, R2)

emitters = [ ("z",e.PLUS(e.NamedAttributeRef("x"), e.NamedAttributeRef("y")))
           , ("w",e.UnnamedAttributeRef(3))
           ]

A = Apply(emitters, J)


myriapart = A

# Use a new "Exec" operator in the logical algebra, that will pass through uninterpreted stuff
sdb = None # connect('http://vega.cs.washington.edu:5555')
aqlscheme = Scheme([("j",INT_TYPE)])
E = ExecScan("SELECT j FROM B WHERE j > 3 AND j < 7", languagetag="aql", connection=sdb, scheme=aqlscheme)

scidbpart = E

exprs = optimize([('A', A), ('E', E)], target=FederatedAlgebra, source=LogicalAlgebra)

print exprs

