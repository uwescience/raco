#!/usr/bin/env python

from raco.scheme import Scheme
from raco.language.federatedlang import FederatedAlgebra, RunAQL, RunMyria
from raco.algebra import Join, Apply, Scan, Sequence, ExecScan, Store
from raco.expression import boolean
from raco.expression import expression as e
from raco.compile import optimize
from raco.types import FLOAT_TYPE, INT_TYPE
from raco.relation_key import RelationKey

from scidbpy import connect

# V0.0: Pass through queries to SciDB

# Construct a conventional RA plan

sch = Scheme([("y",FLOAT_TYPE), ("x",INT_TYPE)])
relkey = RelationKey.from_string("public:adhoc:X")
R1 = Scan(relkey, sch)
R2 = Scan(relkey, sch)

J = Join(boolean.EQ(e.UnnamedAttributeRef(0), e.UnnamedAttributeRef(3)), R1, R2)

emitters = [("z",e.PLUS(e.NamedAttributeRef("x"), e.NamedAttributeRef("y"))),
            ("w",e.UnnamedAttributeRef(3))]

_apply = Apply(emitters, J)
myria_query = Store(RelationKey.from_string("public:adhoc:OUTPUT"), _apply)

# Construct a scidb "plan"
sdb = 'http://vega.cs.washington.edu:8080'
aqlscheme = Scheme([("j",INT_TYPE)])
scidb_query = ExecScan("filter(B1, data>0)", languagetag="aql", connection=sdb,
                       scheme=aqlscheme)

# Build a sequence containing both plans; optimize
seq = Sequence([myria_query, scidb_query])

exprs = optimize(seq, target=FederatedAlgebra())
assert isinstance(exprs, Sequence)

# Separate out the scidb queries for execution
scidb_ops = [x for x in exprs.args if isinstance(x, RunAQL)]
for sop in scidb_ops:
    print repr(sop)
    sdb = connect(sop.connection)
    res = sdb._execute_query(sop.command, response=True)
    print res
