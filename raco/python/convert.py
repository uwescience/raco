# coding=utf-8
""" Functions to convert Python functions to RACO expressions """

import ast

from raco.python.exceptions import PythonSyntaxException, \
    PythonConvertException
from raco.python.util import visitor
from raco.python.util.decompile import get_source


def convert(source_or_ast_or_callable, schema, udfs=None):
    """
    Convert a Python function into its RACO equivalent
    :param source_or_ast_or_callable: Source string, callable, or AST node
    :param schema: List of schema for the input parameter(s)
    :param udfs: List of (name, arity) pairs of UDFs
    :return: RACO expression representing the source, callable, or AST node
    """
    if isinstance(source_or_ast_or_callable, basestring):
        try:
            return convert(ast.parse(source_or_ast_or_callable), schema, udfs)
        except SyntaxError as e:
            raise PythonSyntaxException(e.msg, e.lineno, e.offset)
    elif callable(source_or_ast_or_callable):
        return convert(get_source(source_or_ast_or_callable), schema, udfs)
    elif isinstance(source_or_ast_or_callable, ast.AST):
        return visitor.ExpressionVisitor(schema or [], udfs or []).visit(
            source_or_ast_or_callable) or None
    else:
        raise PythonConvertException(
            'Argument was not a source string, callable, or AST node')
