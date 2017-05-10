"""
A library of expressions that can be composed of existing expressions.
"""

from .udf import Function
import raco.expression
from raco.expression import *


def is_defined(function_name):
    return function_name.lower() in EXPRESSIONS


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


def create_variable_length_function(num_args, func):
    var = ["x{i}".format(i=i + 1) for i in xrange(num_args)]
    var_refs = [NamedAttributeRef(vstr) for vstr in var]
    return Function(var, func(var_refs))


# mapping from name -> dict or Function
# the dict is a mapping from arity -> Function
EXPRESSIONS_CASE = {
    'SafeDiv': {
        2: Function(['n', 'd'], Case(
            [(EQ(NamedAttributeRef('d'), NumericLiteral(0)),
              NumericLiteral(0.0))],
            DIVIDE(NamedAttributeRef('n'), NamedAttributeRef('d')))),
        3: Function(['n', 'd', 'default'], Case(
            [(EQ(NamedAttributeRef('d'), NumericLiteral(0)),
              CAST(types.DOUBLE_TYPE, NamedAttributeRef('default')))],
            DIVIDE(NamedAttributeRef('n'), NamedAttributeRef('d'))))
    },
    'TheAnswerToLifeTheUniverseAndEverything': Function(
        [], NumericLiteral(42)),
    'greatest': lambda num_args: create_nested_binary(num_args, GREATER),
    'least': lambda num_args: create_nested_binary(num_args, LESSER),
    'greater': create_nested_binary(2, GREATER),
    'lexmin': lambda num_args:
        create_variable_length_function(num_args, LEXMIN),
    'lesser': create_nested_binary(2, LESSER),
    'substr': Function(['str', 'begin', 'end'],
                       SUBSTR([NamedAttributeRef('str'),
                               NamedAttributeRef('begin'),
                               NamedAttributeRef('end')
                               ])),
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
                             ])),
    'flip': Function(['p'], LT(RANDOM(), NamedAttributeRef('p')))
}


def get_arity(func_class):
    """Return the arity of built-in Myria expressions."""

    if issubclass(func_class, ZeroaryOperator):
        return 0
    elif issubclass(func_class, UnaryOperator):
        return 1
    elif issubclass(func_class, BinaryOperator):
        return 2
    else:
        # Don't handle n-ary functions automatically
        assert False


def one_to_one_function(func_name):
    """Emit a Function object that wraps a Myria built-in expression."""
    func_class = getattr(raco.expression, func_name)
    arity = get_arity(func_class)
    function_args = ['arg%d' % i for i in range(arity)]
    expression_args = [NamedAttributeRef(x) for x in function_args]
    return Function(function_args, func_class(*expression_args))

# Simple functions that map to a single Myria expression; the names here
# must match the corresponding function class in raco.expression.function
ONE_TO_ONE_FUNCS = ['ABS', 'CEIL', 'COS', 'FLOOR', 'LOG', 'SIN', 'SQRT',
                    'TAN', 'LEN', 'POW', 'MAX', 'MIN', 'SUM', 'AVG', 'STDEV',
                    'COUNTALL', 'MD5', 'RANDOM', 'YEAR', 'MONTH', 'DAY',
                    'SPLIT', 'SEQUENCE', 'NGRAM']

ONE_TO_ONE_EXPRS = {k.lower(): one_to_one_function(k) for k in ONE_TO_ONE_FUNCS}  # noqa

EXPRESSIONS = {k.lower(): v for k, v in EXPRESSIONS_CASE.items()}
EXPRESSIONS.update(ONE_TO_ONE_EXPRS)
