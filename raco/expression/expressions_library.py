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

    var = ["x{i}".format(i=i + 1) for i in xrange(num_args)]
    var_refs = [NamedAttributeRef(vstr) for vstr in var]
    return Function(var, reduce(func, var_refs))


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
    'len': Function(['str'], LEN(NamedAttributeRef('str'))),
    'head': Function(['str', 'length'],
                     SUBSTR([NamedAttributeRef('str'),
                             NumericLiteral(0),
                             LESSER(LEN(NamedAttributeRef('str')),
                                    NamedAttributeRef('length'))
                             ])),
    'tail': Function(['str', 'length'],
                     SUBSTR([NamedAttributeRef('str'),
                             GREATER(MINUS(LEN(NamedAttributeRef('str')),
                                           NamedAttributeRef('length')),
                                     NumericLiteral(0)),
                             LEN(NamedAttributeRef('str'))
                             ]))
}

# Mapping from source symbols to raco.expression.UnaryOperator classes
UNARY_FUNCS = {
    'ABS': ABS,
    'CEIL': CEIL,
    'COS': COS,
    'FLOOR': FLOOR,
    'LOG': LOG,
    'SIN': SIN,
    'SQRT': SQRT,
    'TAN': TAN,
}


def create_unary_function(func_class):
    return Function(['x'], func_class(NamedAttributeRef('x')))

UNARY_EXPRESSIONS = {k.lower(): create_unary_function(v)
                    for k, v in UNARY_FUNCS.iteritems()}  # noqa

EXPRESSIONS = {k.lower(): v for k, v in EXPRESSIONS_CASE.items()}
EXPRESSIONS.update(UNARY_EXPRESSIONS)
