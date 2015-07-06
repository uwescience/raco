# These imports are required here -- for eval inside replace_with_repr
from raco.expression import *
from raco.algebra import *
from raco.relation_key import *
from raco.scheme import *
from raco.backends.myria import *
from raco.backends.cpp import *
from raco.backends.radish import *
from raco.backends.sparql import *

# NOTES: relying on import * for eval is error prone due
#        to namespace collisions
# NOTES: what to do if a operator has two constructors?


def replace_with_repr(plan):
    r = repr(plan)
    try:
        return eval(r)
    except (TypeError, AttributeError, SyntaxError):
        print 'Error with repr {r} of plan {p}'.format(r=r, p=plan)
        raise
