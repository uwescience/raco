# import all the expressions and algebras

from raco.language.myrialang import *
from raco.language.clang import *
from raco.language.clangcommon import *
from raco.language.grappalang import *
from raco.algebra import *
from raco.scheme import *
from raco.expression.expression import *
from raco.expression.aggregate import *
from raco.types import *
from raco.relation_key import *
from raco.expression.boolean import *


import logging
logging.basicConfig()
_LOG = logging.getLogger(name=__name__)


def plan_from_repr(repr_string):
    _LOG.warning("Relying on eval! "
                 "This module should only be used in "
                 "trusted development situations\n")
    return eval(repr_string)
