# coding=utf-8
""" Python AST visitor that converts an expression to a RACO equivalent """

import sys
import ast

from raco.expression import NamedAttributeRef, UnnamedAttributeRef, \
    StringLiteral, NumericLiteral, BooleanLiteral, \
    EQ, NEQ, LT, LTEQ, GT, GTEQ, AND, OR, NOT, \
    PLUS, MINUS, DIVIDE, IDIVIDE, MOD, TIMES, NEG, CAST, PythonUDF
from raco.types import STRING_TYPE, LONG_TYPE, DOUBLE_TYPE, BOOLEAN_TYPE
from raco.expression.function import WORKERID, RANDOM, \
    ABS, CEIL, COS, FLOOR, LOG, SIN, SQRT, TAN, MD5, LEN, POW, \
    LESSER, GREATER, SUBSTR
from raco.python.exceptions import PythonUnrecognizedTokenException, \
    PythonOutOfRangeException, PythonUnsupportedOperationException, \
    PythonArgumentException

comparator_map = {
    ast.Eq: EQ,
    ast.NotEq: NEQ,
    ast.Lt: LT,
    ast.LtE: LTEQ,
    ast.Gt: GT,
    ast.GtE: GTEQ
}

propositional_map = {
    ast.And: AND,
    ast.Or: OR
}

unary_operators = {
    ast.Not: NOT,
    ast.USub: NEG
}

binary_operators = {
    ast.Add: PLUS,
    ast.Sub: MINUS,
    ast.Div: DIVIDE,
    ast.FloorDiv: IDIVIDE,
    ast.Mod: MOD,
    ast.Mult: TIMES
}

literal_map = {
    'True': BooleanLiteral(True),
    'False': BooleanLiteral(False),
}

nary_map = {
    # Arity, name: function
    (1, 'str'): lambda args: CAST(STRING_TYPE, args[0]),
    (1, 'int'): lambda args: MOD(CAST(LONG_TYPE, args[0]),
                                 NumericLiteral(sys.maxint)),
    (1, 'long'): lambda args: CAST(LONG_TYPE, args[0]),
    (1, 'float'): lambda args: CAST(DOUBLE_TYPE, args[0]),
    (1, 'bool'): lambda args: CAST(BOOLEAN_TYPE, args[0]),

    (0, 'workerid'): lambda args: WORKERID(),
    (0, 'random'): lambda args: RANDOM(),
    (1, 'fabs'): lambda args: ABS(CAST(DOUBLE_TYPE, args[0])),
    (1, 'abs'): lambda args: ABS(args[0]),
    (1, 'ceil'): lambda args: CEIL(args[0]),
    (1, 'cos'): lambda args: COS(args[0]),
    (1, 'floor'): lambda args: FLOOR(args[0]),
    (1, 'log'): lambda args: LOG(args[0]),
    (1, 'sin'): lambda args: SIN(args[0]),
    (1, 'sqrt'): lambda args: SQRT(args[0]),
    (1, 'tan'): lambda args: TAN(args[0]),
    (1, 'md5'): lambda args: MD5(args[0]),
    (1, 'len'): lambda args: LEN(args[0]),
    (2, 'pow'): lambda args: POW(args[0], args[1]),

    (2, 'min'): lambda args: LESSER(args[0], args[1]),
    (2, 'max'): lambda args: GREATER(args[0], args[1]),
}

zero = ast.Num(n=0)


class ExpressionVisitor(ast.NodeVisitor):
    """ Visitor that converts an AST to a RACO expression """

    def __init__(self, schema, udfs):
        self.schema = schema
        self.names = None
        self.udfs = udfs

    def visit_arguments(self, node):
        """ Visitor for function arguments """
        self.names = [n.id for n in node.args]

    def visit_UnaryOp(self, node):
        """ Visitor for a unary operator """
        if type(node.op) not in unary_operators:
            raise PythonUnrecognizedTokenException(node.op,
                                                   node.lineno,
                                                   node.col_offset)
        return unary_operators[type(node.op)](self.visit(node.operand))

    def visit_BinOp(self, node):
        """ Visitor for binary operations """
        if type(node.op) not in binary_operators:
            raise PythonUnrecognizedTokenException(node.op,
                                                   node.lineno,
                                                   node.col_offset)
        return binary_operators[type(node.op)](
            self.visit(node.left), self.visit(node.right))

    def visit_BoolOp(self, node):
        """ Visitor for boolean operations """
        assert (len(node.values) >= 2)
        op = propositional_map[type(node.op)]
        return reduce(lambda c, e: op(c, self.visit(e)),
                      # Fold over any other clauses
                      node.values[2:],
                      # Always at least two clauses
                      op(*map(self.visit, node.values[:2])))

    def visit_Compare(self, node):
        """ Visitor for comparison operations """
        if len(node.ops) == 1 and \
           type(node.ops[0]) in comparator_map.keys() and \
           len(node.comparators) == 1:
            left = self.visit(node.left)
            right = self.visit(node.comparators[0])
            return comparator_map[type(node.ops[0])](left, right)
        else:
            raise PythonUnrecognizedTokenException(node.ops[0],
                                                   node.lineno,
                                                   node.col_offset)

    def visit_Attribute(self, node):
        """ Visitor for dotted references """
        if not isinstance(node.value, ast.Name):
            raise PythonUnsupportedOperationException(
                'Unsupported reference',
                node.lineno, node.col_offset)

        scheme = self.schema[self.names.index(node.value.id)]

        if node.attr not in [s[0] for s in scheme]:
            raise PythonUnrecognizedTokenException(node.attr,
                                                   node.lineno,
                                                   node.col_offset)

        return NamedAttributeRef(node.attr)

    def visit_Subscript(self, node):
        """ Visitor for slices """
        if isinstance(node.value, ast.Name) and \
           isinstance(node.slice, ast.Index) and \
           node.value.id in self.names:
            return self.visit_AttributeIndex(node)
        elif isinstance(node.slice, ast.Slice):
            return self.visit_SubstringSlice(node)
        elif isinstance(node.slice, ast.Index):
            return self.visit_SubstringIndex(node)

    def visit_AttributeIndex(self, node):
        """ Visitor for subscripts over a tuple """
        if node.value.id not in self.names:
            raise PythonUnrecognizedTokenException(
                node.value.id, node.lineno, node.col_offset)

        offset = sum([len(s) for s in
                      self.schema[:self.names.index(node.value.id)]])

        if node.slice.value.n < 0 or \
           node.slice.value.n >= \
                len(self.schema[self.names.index(node.value.id)]):
            raise PythonOutOfRangeException(
                node.value.id, node.lineno, node.col_offset)

        return UnnamedAttributeRef(node.slice.value.n + offset)

    def visit_SubstringIndex(self, node):
        """ Visitor for indexing a string """
        node.slice = ast.Slice(lower=node.slice.value,
                               upper=ast.Num(node.slice.value.n + 1),
                               step=None)
        return self.visit_SubstringSlice(node)

    def visit_SubstringSlice(self, node):
        """ Visitor for slicing a string """
        if (node.slice.lower or zero).n < 0 or \
           (node.slice.upper or zero).n < 0:
            raise PythonUnsupportedOperationException(
                'RACO does not support negative indices in slices',
                node.lineno, node.col_offset)
        elif node.slice.step is not None:
            raise PythonUnsupportedOperationException(
                'RACO does not support steps in slices',
                node.lineno, node.col_offset)

        child = self.visit(node.value)
        return SUBSTR(
            [child,
             self.visit(node.slice.lower or ast.Num(0)),
             self.visit(node.slice.upper or ast.Num(2 ** 30))])

    def visit_Call(self, node):
        """ Visitor for calling built-in or UDF functions """
        name = node.func.id
        arity = len(node.args)

        if (arity, name) in nary_map:
            return self.visit_Call_builtin(node)
        elif name in (udf['name'] for udf in self.udfs):
            return self.visit_Call_UDF(node)
        else:
            raise PythonArgumentException(
                'Unrecognized function %s or invalid arguments' % name,
                node.lineno, node.col_offset)

    def visit_Call_builtin(self, node):
        """ Visitor for calling built-in functions """
        name = node.func.id
        arity = len(node.args)

        if (arity, name) not in nary_map:
            raise PythonArgumentException(
                'Unrecognized function %s or invalid arguments' % name,
                node.lineno, node.col_offset)

        return nary_map[(arity, name)](map(self.visit, node.args))

    def visit_Call_UDF(self, node):
        """ Visitor for calling UDF functions """
        name = node.func.id
        arity = len(node.args)
        udf = next((udf for udf in self.udfs
                    if udf['name'] == name), None)

        # Ignore output type when looking up UDF
        if udf is None:
            raise PythonArgumentException(
                'Unrecognized function %s or invalid arguments' % name,
                node.lineno, node.col_offset)

        output_type = udf['outputType']
        source = udf.get('source', None)
        return PythonUDF(name, output_type,
                         *map(self.visit, node.args),
                         source=source)

    def visit_Str(self, node):
        """ Visitor for string literals """
        return StringLiteral(node.s)

    def visit_Num(self, node):
        """ Visitor for numeric literals """
        return NumericLiteral(node.n)

    def visit_Name(self, node):
        """ Visitor for built-in literals """
        if node.id not in literal_map:
            raise PythonUnrecognizedTokenException(node.id,
                                                   node.lineno,
                                                   node.col_offset)
        return literal_map[node.id]

    def visit_Module(self, node):
        """ Visitor for top-level modules """
        assert(len(node.body) == 1)
        return self.visit(node.body[0])

    def visit_Lambda(self, node):
        """ Visitor for lambdas """
        self.visit(node.args)
        return self.visit(node.body)

    def visit_FunctionDef(self, node):
        """ Visitor for function declaration """
        # TODO this is not a strict RACO requirement
        if len(node.body) != 1:
            raise PythonUnsupportedOperationException(
                'Functions must have exactly one statement',
                node.lineno, node.col_offset)
        elif not isinstance(node.body[0], ast.Return):
            raise PythonUnsupportedOperationException(
                'Statement in function body must be a return',
                node.lineno, node.col_offset)

        self.visit(node.args)
        return self.visit(node.body[0])

    def visit_Return(self, node):
        """ Visitor for return statements """
        return self.visit(node.value)

    def visit_Expr(self, node):
        """ Visitor for expressions """
        return self.visit(node.value)

    def generic_visit(self, node):
        """ Visitor for unsupported node types """
        raise PythonUnsupportedOperationException(
            'Unsupported node ' + str(type(node)),
            node.lineno, node.col_offset)
