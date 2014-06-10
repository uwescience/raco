import raco.pyra as pyra
import raco.scheme as scheme
import raco.catalog

sch = pyra.Scheme([("subject", "LONG_TYPE"), ("predicate", "LONG_TYPE"), ("object", "LONG_TYPE")])
test = pyra.Relation(sch, [(1,2,3),(1,3,4),(2,2,8),(2,3,9)])


data = [
(1,'knows',2),
(1,'knows',3),
(2,'holdsAccount',4),
(3,'holdsAccount',5),
(8,'holdsAccount',9),
(4,'accountServiceHomepage',6),
(5,'accountServiceHomepage',6)
]

sch = pyra.Scheme([("subject", "LONG_TYPE"), ("predicate", "STRING_TYPE"), ("object", "LONG_TYPE")])
Rr = pyra.Relation(sch, data)

btc_schema = {
  "trial" : raco.catalog.ASCIIFile("trial.dat", pyra.Scheme([("subject", "LONG_TYPE"), ("predicate", "LONG_TYPE"), ("object", "LONG_TYPE")])),
  "btc2010" : raco.catalog.ASCIIFile("btc2010.dat", pyra.Scheme([("subject", "LONG_TYPE"), ("predicate", "LONG_TYPE"), ("object", "LONG_TYPE")])),
}
