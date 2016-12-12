# coding=utf-8
""" Utilities for converting compiled functions to Python source """

import ast
import inspect
import re
from types import CodeType

from raco.python.exceptions import PythonConvertException

lambda_header = 'lambda:'        # Prefix associated with a lambda
module_name = '<module>'         # Name of a non-lambda code unit
min_length = len(lambda_header)  # shortest lambda expression possible
lambda_expression = r'(?:^|\W)(?=(lambda[^:]*:.*))'
function_expression = r'(?:^|\W)(?=(def\W[^:]*:.*))'


def get_source(f):
    """ Return the source of a lambda function """
    try:
        lines, _ = inspect.getsourcelines(f)
    except (IOError, TypeError) as e:
        raise PythonConvertException(str(e))

    source = ' '.join(line.strip(' ') for line in lines).strip('\n\r ')
    candidates = _get_candidates(source)

    try:
        return next(source for candidate in candidates
                    for source in [_get_source(f, candidate)]
                    if source)
    except StopIteration:
        raise PythonConvertException('Unable to convert function into source')


def _get_source(f, source):
    """
    Convert the given source into an AST
    :param source: Source code of a function or lambda expression
    :return: An AST representing the source
    """
    for source in _reverse_walk_source(source):
        try:
            tree = ast.parse(source)
            node = _get_definition_node(tree)
            return _verify_ast(f, node, source)
        except SyntaxError:
            pass

    return None


def _verify_ast(f, tree, source):
    """
    Verify that a given tree matches the original function
    AST nodes maintain their starting offsets in the input source,
    so we have to work backwards to determine the actual end.

    :param tree: an AST representation of a function or lambda
    :return: True when the AST matches the function or lambda; false otherwise
    """

    for source, body in zip(
            _reverse_walk_source(source[tree.col_offset:], min_length),
            _reverse_walk_source(_get_body_source(tree, source))):
        try:
            recompiled_f = _compile(source, body)
            if _verify_bytecode(source, f, recompiled_f):
                return source
        except SyntaxError:
            pass

    return None


def _compile(source, body):
    try:
        return compile(body, '', 'eval')
    except SyntaxError:
        module = compile(source, '', 'exec')
        function = filter(None, module.co_consts)[-1]  # Best guess
        if module.co_name != module_name:
            raise PythonConvertException('Function must compile to a module')
        elif not isinstance(function, CodeType):
            raise PythonConvertException('Module must contain function')
        return function


def _get_body_source(tree, source):
    first_statement = tree.body[0] if isinstance(tree.body, list) \
        else tree.body
    line_offset = sum(len(line) for line
                      in source.split('\n')[:first_statement.lineno - 1])
    return source[line_offset + first_statement.col_offset - 1:].strip(': ')


def _verify_bytecode(source, f, recompiled_f, require_callable=True):
    try:
        return ((not require_callable or
                 callable(eval(source))) and
                filter(None, f.func_code.co_consts) ==
                filter(None, recompiled_f.co_consts) and
                len(f.__code__.co_code) == len(recompiled_f.co_code))
    except NameError:
        return False
    except SyntaxError as e:
        if not source.startswith(lambda_header):
            return _verify_bytecode(source, f, recompiled_f, False) \
                if isinstance(recompiled_f, CodeType) \
                else False
        else:
            raise e


def _reverse_walk_source(source, min_length=0):
    while len(source) > min_length:
        yield source
        source = source[:-1].strip()


def _get_candidates(source):
    """
    Get all possible lambda expressions in a source string,
    where the results may contain incorrect trailing characters
    (we'll remove these later)

    E.g.: lambda x: 'lambda:0' -> ["lambda x: 'lambda:0'", "lambda:0'"]
    """
    lambdas = re.findall(lambda_expression, source, re.MULTILINE | re.DOTALL)
    functions = re.findall(function_expression, source,
                           re.MULTILINE | re.DOTALL)
    return lambdas + functions


def _get_definition_node(tree):
    return next((node for node in ast.walk(tree) if
                 isinstance(node, ast.Lambda) or
                 isinstance(node, ast.FunctionDef)))
