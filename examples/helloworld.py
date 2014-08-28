from raco.compile import compile, optimize
from raco.expression.boolean import EQ, AND, OR
from raco.expression import NamedAttributeRef, StringLiteral, NumericLiteral
import raco.scheme
import raco.catalog

# declare the schema for each relation
sch = raco.scheme.Scheme([("subject", int), ("predicate", int), ("object", int)])

# Create a relation object.  We can add formats here as needed.
trialdat = raco.catalog.ASCIIFile("trial.dat", sch)
print sch

# Now write the RA expression

# Scan just takes a pointer to a relation object
R = Scan(trialdat, sch)  #TODO: is this supposed to pass sch?
print R.scheme()


# Select
# EQ(x,y) means x=y, GT(x,y) means x>y, etc.
sR = Select(EQ(NamedAttributeRef("predicate"), NumericLiteral(1133564893)), R)
sS = Select(EQ(NamedAttributeRef("predicate"), NumericLiteral(77645021)), R)
#sT = Select(EQ(NamedAttributeRef("predicate"), NumericLiteral(77645021)), R)
sT = Select(EQ(NamedAttributeRef("object"), NumericLiteral(1018848684)), R)

# Join([(w,x),(y,z)], R, S) means "JOIN R, S ON (R.w = S.x AND R.y = S.z)"
sRsS = Join([("object","subject")], sR, sS)
sRsSsT = Join([("object","subject")], sRsS, sT)

# optimize applies a set of rules to translate a source
# expression to a target expression
result = optimize([("Ans", sT)], CCAlgebra)

# compile generates the linear code from the expression tree
print compile(result)
