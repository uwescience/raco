'''
A library of expressions that can be composed of existing expressions.
'''

from .udf import Function
from raco.expression import *


def lookup(function_name, num_args):
    func = EXPRESSIONS.get(function_name.lower())
    if hasattr(func, '__call__'):
        return func(num_args)
    if isinstance(func, dict):
        return func.get(num_args)
    return func


def create_nested_binary(num_args, func):
    if num_args < 2:
        return None

    def make_binary(num_args):
        if num_args == 2:
            return func(
                NamedAttributeRef('x%d' % num_args),
                NamedAttributeRef('x%d' % (num_args - 1)))
        return func(
            NamedAttributeRef('x%d' % num_args),
            make_binary(num_args - 1))
    return Function(
        ['x' + str(x + 1) for x in range(num_args)],
        make_binary(num_args))


# mapping from name -> dict or Function
# the dict is a mapping from arity -> Function
EXPRESSIONS_CASE = {
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
    'TheAnswerToLifeTheUniverseAndEverything': Function(
        [], NumericLiteral(42)),
    'greatest': lambda num_args: create_nested_binary(num_args, GREATER),
    'least': lambda num_args: create_nested_binary(num_args, LESSER),
    'greater': create_nested_binary(2, GREATER),
    'lesser': create_nested_binary(2, LESSER),
    'substr': Function(['str', 'begin', 'end'],
                       SUBSTR([NamedAttributeRef('str'),
                               NamedAttributeRef('begin'),
                               NamedAttributeRef('end')
                               ])),
    'len': Function(['str'], LEN(NamedAttributeRef('str')))
}

EXPRESSIONS = {k.lower(): v for k, v in EXPRESSIONS_CASE.items()}
