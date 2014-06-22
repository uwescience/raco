from raco.scheme import Scheme
from raco.language import FederatedAlgebra
from raco.algebra import *
from raco.expression import boolean
from raco.expression import expression as e
from raco.compile import optimize
from raco.types import FLOAT_TYPE, INT_TYPE
from raco.relation_key import RelationKey
import raco

from scidbpy import connect, SciDBQueryError, SciDBArray

sch = Scheme([("y",FLOAT_TYPE), ("x",INT_TYPE)])

# V0.0: Pass through queries to SciDB

# Construct a conventional RA plan
R = RelationKey.from_string("public:adhoc:X")

R1 = Scan(R, sch)
R2 = Scan(R, sch)

J = Join(boolean.EQ(e.UnnamedAttributeRef(0), e.UnnamedAttributeRef(3)), R1, R2)

emitters = [ ("z",e.PLUS(e.NamedAttributeRef("x"), e.NamedAttributeRef("y")))
           , ("w",e.UnnamedAttributeRef(3))
           ]

_apply = Apply(emitters, J)
myria_query = Store(RelationKey.from_string("public:adhoc:OUTPUT"), _apply)

sdb = None # connect('http://vega.cs.washington.edu:5555')
aqlscheme = Scheme([("j",INT_TYPE)])
scidb_query = ExecScan("SELECT j FROM B WHERE j > 3 AND j < 7",
                        languagetag="aql", connection=sdb, scheme=aqlscheme)

seq = Sequence([myria_query, scidb_query])

exprs = optimize(seq, target=FederatedAlgebra(), source=LogicalAlgebra)
for i, op in enumerate(exprs.args):
    print "[%d] %r\n" % (i, op)

