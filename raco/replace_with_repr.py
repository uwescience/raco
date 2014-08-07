# Not really unused -- for eval inside replace_with_repr
from raco.expression import *
from raco.algebra import *
from raco.relation_key import *
from raco.scheme import *
from raco.language.myrialang import *


def replace_with_repr(plan):
    r = repr(plan)
    try:
        return eval(r)
    except (TypeError, AttributeError):
        print 'Error with repr {r} of plan {p}'.format(r=r, p=plan)
        raise