'''
A library of expressions that can be composed of existing expressions.
'''

from .udf import Function
from raco.expression import *


def lookup(function_name, num_args):
    func = EXPRESSIONS.get(function_name)
    if isinstance(func, dict):
        func = func.get(num_args)
    return func

# mapping from name -> dict or Function
# the dict is a mapping from arity -> Function
EXPRESSIONS = {
    'SafeDiv': {
        2: Function(['n', 'd'], Case(
            [(EQ(NamedAttributeRef('d'), NumericLiteral(0)),
              NumericLiteral(0))],
            DIVIDE(NamedAttributeRef('n'), NamedAttributeRef('d')))),
        3: Function(['n', 'd', 'default'], Case(
            [(EQ(NamedAttributeRef('d'), NumericLiteral(0)),
              NamedAttributeRef('default'))],
            DIVIDE(NamedAttributeRef('n'), NamedAttributeRef('d'))))
    },
    'TheAnswerToLifeUniverseAndEverything': Function([], NumericLiteral(42))
}
