# -*- coding: UTF-8 -*-

import collections
import sys

from ply import yacc

from raco import relation_key
import raco.myrial.scanner as scanner
import raco.scheme as scheme
import raco.types
import raco.expression as sexpr
import raco.myrial.emitarg as emitarg
from raco.expression.udf import Function, StatefulFunc
import raco.expression.expressions_library as expr_lib
from .exceptions import *
import raco.types
from raco.expression import StateVar, PythonUDF, UnnamedAttributeRef, \
    VariadicFunction


class JoinColumnCountMismatchException(Exception):
    pass

# ID is a symbol name that identifies an input expression; columns is a list of
# columns expressed as either names or integer positions.
JoinTarget = collections.namedtuple('JoinTarget', ['expr', 'columns'])

SelectFromWhere = collections.namedtuple(
    'SelectFromWhere', ['distinct', 'select', 'from_', 'where', 'limit'])

DecomposableAgg = collections.namedtuple(
    'DecomposableAgg', ['logical', 'local', 'remote'])

# Mapping from source symbols to raco.expression.BinaryOperator classes
binops = {
    '+': sexpr.PLUS,
    '-': sexpr.MINUS,
    '/': sexpr.DIVIDE,
    '//': sexpr.IDIVIDE,
    '%': sexpr.MOD,
    '*': sexpr.TIMES,
    '>': sexpr.GT,
    '<': sexpr.LT,
    '>=': sexpr.GTEQ,
    u'≥': sexpr.GTEQ,
    '<=': sexpr.LTEQ,
    u'≤': sexpr.LTEQ,
    '!=': sexpr.NEQ,
    '<>': sexpr.NEQ,
    u'≠': sexpr.NEQ,
    '==': sexpr.EQ,
    '=': sexpr.EQ,
    'AND': sexpr.AND,
    'OR': sexpr.OR,
    'LIKE': sexpr.LIKE
}

# Map from myrial token name to raco internal type name.
myrial_type_map = {
    "STRING": raco.types.STRING_TYPE,
    "INT": raco.types.LONG_TYPE,
    "FLOAT": raco.types.DOUBLE_TYPE,
    "BOOLEAN": raco.types.BOOLEAN_TYPE,
    "BLOB": raco.types.BLOB_TYPE
}


def contains_tuple_expression(ex):
    """Return True if an Expression contains a TupleExpression"""
    return any(isinstance(sx, TupleExpression) for sx in ex.walk())


def check_no_tuple_expression(ex, lineno):
    if contains_tuple_expression(ex):
        raise NestedTupleExpressionException(lineno)


def check_simple_expression(ex, lineno):
    check_no_tuple_expression(ex, lineno)
    sexpr.check_no_aggregate(ex, lineno)


def get_emitters(ex):
    if isinstance(ex, TupleExpression):
        return ex.emitters
    else:
        return [ex]


def get_num_emitters(ex):
    return len(get_emitters(ex))


class TupleExpression(sexpr.Expression):
    """Represents an instance of a tuple-valued Expression

    This class is a pseudo-expression that corresponds to a UDA or stateful
    apply with multiple return values.  Myria doesn't support tuples as a
    first-class data type.  Instead, instances of TupleExpression are converted
    into multiple scalar expression instances.
    """
    def __init__(self, emitters):
        self.emitters = emitters

    def walk(self):
        yield self
        for emitter in self.emitters:
            for x in emitter.walk():
                yield x

    def apply(self, f):
        self.emitters = [f(e) for e in self.emitters]

    def check_for_nested(self, lineno):
        """Raise an exception if a sub-expression contains a TupleExpression"""

        for ex in self.emitters:
            check_no_tuple_expression(ex, lineno)

    def get_children(self):
        return self.emitters

    def typeof(self, scheme, state_scheme):
        """Type checks are not applied to TupleExpressions."""
        raise NotImplementedError()

    def evaluate(self, _tuple, scheme, state=None):
        raise NotImplementedError()


class Parser(object):
    # mapping from function name to Function tuple
    udf_functions = {}

    # state modifier variables accessed by the current emit argument
    statemods = []

    # A unique ID pool for the stateful apply state variables
    mangle_id = 0

    # mapping from UDA name to local, remote aggregates
    decomposable_aggs = {}

    def __init__(self, log=yacc.PlyLogger(sys.stderr)):
        self.log = log
        self.tokens = scanner.tokens

        # Precedence among scalar expression operators in ascending order; this
        # is necessary to disambiguate the grammar.  Operator precedence is
        # identical to Python:
        # http://docs.python.org/2/reference/expressions.html#comparisons

        self.precedence = (
            ('left', 'OR'),
            ('left', 'AND'),
            ('right', 'NOT'),
            ('left', 'EQ', 'EQUALS', 'NE', 'GT', 'LT', 'LE', 'GE', 'LIKE'),
            ('left', 'PLUS', 'MINUS'),
            ('left', 'TIMES', 'DIVIDE', 'IDIVIDE', 'MOD'),
            ('right', 'UMINUS'),    # Unary minus
        )

    # A MyriaL program consists of 1 or more "translation units", each of which
    # is a function, apply definition, or statement.
    @staticmethod
    def p_translation_unit_list(p):
        """translation_unit_list : translation_unit_list translation_unit
                                 | translation_unit"""
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_translation_unit(p):
        """translation_unit : statement
                            | constant
                            | udf
                            | apply
                            | uda
                            | decomposable_uda"""
        p[0] = p[1]

    @staticmethod
    def check_for_undefined(p, name, _sexpr, args):
        undefined = sexpr.udf_undefined_vars(_sexpr, args)
        if undefined:
            raise UndefinedVariableException(name, undefined[0], p.lineno(0))

    @staticmethod
    def check_for_reserved(p, name):
        """Check whether an identifier name is reserved."""
        if expr_lib.is_defined(name):
            raise ReservedTokenException(name, p.lineno(0))

    @staticmethod
    def add_decomposable_uda(p, logical, local, remote):
        """Register a decomposable UDA.

        :param p: The parser context
        :param logical: The name of the logical UDA
        :param local: The name of the local UDA
        :param remote: The name of the remote UDA
        """
        lineno = p.lineno(0)

        if logical in Parser.decomposable_aggs:
            raise DuplicateFunctionDefinitionException(logical, lineno)

        def check_name(name):
            if name not in Parser.udf_functions:
                raise NoSuchFunctionException(lineno)

            func = Parser.udf_functions[name]
            if not isinstance(func, StatefulFunc):
                raise NoSuchFunctionException(lineno)
            if not sexpr.expression_contains_aggregate(func.sexpr):
                raise NoSuchFunctionException(lineno)
            return func

        da = DecomposableAgg(*[check_name(x) for x in
                             (logical, local, remote)])

        # Do some basic sanity checking of the arguments; we can't do full
        # type inspection here, because the full type information is not
        # known until the function is invoked.

        # Number of local inputs must match number of logical inputs
        if len(da.local.args) != len(da.logical.args):
            raise InvalidArgumentList(local, da.logical.args, lineno)

        # Number of local outputs must equal number of remote inputs
        num_local_emitters = get_num_emitters(da.local.sexpr)
        if num_local_emitters != len(da.remote.args):
            phony_names = ['x%d' % n for n in range(num_local_emitters)]
            raise InvalidArgumentList(remote, phony_names, lineno)

        # Number of remote outputs must match number of logical outputs
        if get_num_emitters(da.logical.sexpr) != get_num_emitters(da.remote.sexpr):  # noqa
            raise InvalidEmitList(remote, lineno)

        Parser.decomposable_aggs[logical] = da

    @staticmethod
    def add_nary_udf(p, name, args, emitters):
        """Add an n-ary user-defined function to the global function table.

        :param p: The parser context
        :param name: The name of the function
        :type name: string
        :param args: A list of function arguments
        :type args: list of strings
        :param emitter: The output expression(s)
        :type body_expr: A list of NaryEmitArg instances
        """
        if not all(isinstance(e, emitarg.NaryEmitArg) for e in emitters):
            raise IllegalWildcardException(name, p.lineno(0))
        if sum(len(x.sexprs) for x in emitters) != len(emitters):
            raise NestedTupleExpressionException(p.lineno(0))
        emit_exprs = [e.sexprs[0] for e in emitters]
        Parser.add_udf(p, name, args, emit_exprs)

    @staticmethod
    def add_udf(p, name, args, body_exprs):
        """Add a user-defined function to the global function table.

        :param p: The parser context
        :param name: The name of the function
        :type name: string
        :param args: A list of function arguments
        :type args: list of strings
        :param body_exprs: A list of scalar expressions containing the body
        :type body_exprs: list of raco.expression.Expression
        """
        if name in Parser.udf_functions:
            raise DuplicateFunctionDefinitionException(name, p.lineno(0))

        if len(args) != len(set(args)):
            raise DuplicateVariableException(name, p.lineno(0))

        if len(body_exprs) == 1:
            emit_op = body_exprs[0]
        else:
            emit_op = TupleExpression(body_exprs)

        Parser.check_for_undefined(p, name, emit_op, args)

        Parser.udf_functions[name] = Function(args, emit_op)
        return emit_op

    @staticmethod
    def add_python_udf(name, typ, **kwargs):
        """Add a Python user-defined function to the global function table.

        :param name: The name of the function
        :type name: string
        :param typ: The output type of the function
        :type typ: string
        """
        if name in Parser.udf_functions:
            raise DuplicateFunctionDefinitionException(name, -1)

        f = VariadicFunction(PythonUDF, name, typ, **kwargs)
        Parser.udf_functions[name] = f
        return f

    @staticmethod
    def mangle(name):
        Parser.mangle_id += 1
        return "{name}__{mid}".format(name=name, mid=Parser.mangle_id)

    @staticmethod
    def add_state_func(p, name, args, inits, updates, emitters, is_aggregate):
        """Register a stateful apply or UDA.

        :param name: The name of the function
        :param args: A list of function argument names (strings)
        :param inits: A list of NaryEmitArg that describe init logic; each
        should contain exactly one emit expression.
        :param updates: A list of Expression that describe update logic
        :param emitters: An Expression list that returns the final results.
        If None, all statemod variables are returned in the order specified.
        :param is_aggregate: True if the state_func is a UDA

        TODO: de-duplicate logic from add_udf.
        """
        lineno = p.lineno(0)
        if name in Parser.udf_functions:
            raise DuplicateFunctionDefinitionException(name, lineno)
        if len(args) != len(set(args)):
            raise DuplicateVariableException(name, lineno)
        if len(inits) != len(updates):
            raise BadApplyDefinitionException(name, lineno)

        # Unpack the update, init expressions into a statemod dictionary
        statemods = collections.OrderedDict()
        for init, update in zip(inits, updates):
            if not isinstance(init, emitarg.NaryEmitArg):
                raise IllegalWildcardException(name, lineno)

            if len(init.sexprs) != 1:
                raise NestedTupleExpressionException(lineno)

            # Init, update expressions contain tuples or contain aggregates
            check_simple_expression(init.sexprs[0], lineno)
            check_simple_expression(update, lineno)

            if not init.column_names:
                raise UnnamedStateVariableException(name, lineno)

            # check for duplicate variable definitions
            sm_name = init.column_names[0]
            if sm_name in statemods or sm_name in args:
                raise DuplicateVariableException(name, lineno)

            statemods[sm_name] = (init.sexprs[0], update)

        # Check for undefined variables:
        #  - Init expressions cannot reference any variables.
        #  - Update expression can reference function arguments and state
        #    variables.
        #  - The emitter expressions can reference state variables.
        allvars = statemods.keys() + args
        for init_expr, update_expr in statemods.itervalues():
            Parser.check_for_undefined(p, name, init_expr, [])
            Parser.check_for_undefined(p, name, update_expr, allvars)

        if emitters is None:
            emitters = [sexpr.NamedAttributeRef(v) for v in statemods.keys()]

        for e in emitters:
            Parser.check_for_undefined(p, name, e, statemods.keys())
            check_simple_expression(e, lineno)

        # If the function is a UDA, wrap the output expression(s) so
        # downstream users can distinguish stateful apply from
        # aggregate expressions.
        if is_aggregate:
            emitters = [sexpr.UdaAggregateExpression(e) for e in emitters]

        assert len(emitters) > 0
        if len(emitters) == 1:
            emit_op = emitters[0]
        else:
            emit_op = TupleExpression(emitters)

        Parser.udf_functions[name] = StatefulFunc(args, statemods, emit_op)

    @staticmethod
    def p_unreserved_id(p):
        'unreserved_id : ID'
        Parser.check_for_reserved(p, p[1])
        p[0] = p[1]

    @staticmethod
    def p_unreserved_id_list(p):
        """unreserved_id_list : unreserved_id_list COMMA unreserved_id
                              | unreserved_id"""
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_udf(p):
        """udf : DEF unreserved_id LPAREN optional_arg_list RPAREN COLON sexpr SEMI"""  # noqa
        Parser.add_udf(p, p[2], p[4], [p[7]])
        p[0] = None

    @staticmethod
    def p_nary_udf(p):
        """udf : DEF unreserved_id LPAREN optional_arg_list RPAREN COLON table_literal SEMI"""  # noqa
        Parser.add_nary_udf(p, p[2], p[4], p[7])
        p[0] = None

    @staticmethod
    def p_constant(p):
        """constant : CONST unreserved_id COLON sexpr SEMI"""
        Parser.add_udf(p, p[2], [], [p[4]])
        p[0] = None

    @staticmethod
    def p_optional_arg_list(p):
        """optional_arg_list : function_arg_list
                             | empty"""
        p[0] = p[1] or []

    @staticmethod
    def p_recursion_mode(p):
        """recursion_mode : SYNC
                          | ASYNC
                          | empty"""
        p[0] = p[1] or []

    @staticmethod
    def p_pull_order_policy(p):
        """pull_order_policy : ALTERNATE
                             | PULL_IDB
                             | PULL_EDB
                             | BUILD_EDB
                             | empty"""
        p[0] = p[1] or []

    @staticmethod
    def p_function_arg_list(p):
        """function_arg_list : function_arg_list COMMA unreserved_id
                             | unreserved_id"""
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_statefunc_emit_list(p):
        """statefunc_emit_list : LBRACKET sexpr_list RBRACKET SEMI
                               | sexpr SEMI
                               | empty"""
        if len(p) == 5:
            p[0] = p[2]
        elif len(p) == 3:
            p[0] = (p[1],)
        else:
            p[0] = None

    @staticmethod
    def p_decomposable_uda(p):
        'decomposable_uda : UDA TIMES unreserved_id LBRACE unreserved_id COMMA unreserved_id RBRACE SEMI'  # noqa
        logical = p[3]
        local = p[5]
        remote = p[7]
        Parser.add_decomposable_uda(p, logical, local, remote)
        p[0] = None

    @staticmethod
    def p_uda(p):
        'uda : UDA unreserved_id LPAREN optional_arg_list RPAREN LBRACE \
        table_literal SEMI LBRACKET sexpr_list RBRACKET SEMI statefunc_emit_list RBRACE SEMI'  # noqa

        name = p[2]
        args = p[4]
        inits = p[7]
        updates = p[10]
        emits = p[13]
        Parser.add_state_func(p, name, args, inits, updates, emits, True)
        p[0] = None

    @staticmethod
    def p_apply(p):
        'apply : APPLY unreserved_id LPAREN optional_arg_list RPAREN LBRACE \
        table_literal SEMI LBRACKET sexpr_list RBRACKET SEMI statefunc_emit_list RBRACE SEMI'  # noqa

        name = p[2]
        args = p[4]
        inits = p[7]
        updates = p[10]
        emits = p[13]
        Parser.add_state_func(p, name, args, inits, updates, emits, False)
        p[0] = None

    @staticmethod
    def p_statement_assign(p):
        'statement : unreserved_id EQUALS rvalue SEMI'
        p[0] = ('ASSIGN', p[1], p[3])

    @staticmethod
    def p_idbassign(p):
        'idbassign : unreserved_id EQUALS LBRACKET emit_arg_list \
            RBRACKET LARROW rvalue SEMI'
        p[0] = ('IDBASSIGN', p[1], p[4], p[7])

    @staticmethod
    def p_statement_empty(p):
        'statement : SEMI'
        p[0] = None  # stripped out by parse

    # expressions must be embeddable in other expressions; certain constructs
    # are not embeddable, but are available as r-values in an assignment
    @staticmethod
    def p_rvalue(p):
        """rvalue : expression
                  | select_from_where"""
        p[0] = p[1]

    @staticmethod
    def p_idbassign_list(p):
        """idbassign_list : idbassign_list idbassign
                          | idbassign"""
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_statement_list(p):
        """statement_list : statement_list statement
                          | statement"""
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_statement_dowhile(p):
        'statement : DO statement_list WHILE expression SEMI'
        p[0] = ('DOWHILE', p[2], p[4])

    @staticmethod
    def p_statement_dountilconvergence(p):
        ('statement : DO idbassign_list UNTIL CONVERGENCE '
         'recursion_mode pull_order_policy SEMI')
        p[0] = ('UNTILCONVERGENCE', p[2], p[5], p[6])

    @staticmethod
    def p_statement_store(p):
        'statement : STORE LPAREN unreserved_id COMMA relation_key optional_part_info RPAREN SEMI'  # noqa
        p[0] = ('STORE', p[3], p[5], p[6])

    @staticmethod
    def p_statement_sink(p):
        'statement : SINK LPAREN unreserved_id RPAREN SEMI'  # noqa
        p[0] = ('SINK', p[3])

    @staticmethod
    def p_statement_dump(p):
        'statement : DUMP LPAREN unreserved_id RPAREN SEMI'
        p[0] = ('DUMP', p[3])

    @staticmethod
    def p_optional_part_info(p):
        """optional_part_info : COMMA LBRACKET column_ref_list RBRACKET
                              | COMMA HASH LPAREN column_ref_list RPAREN
                              | COMMA BROADCAST LPAREN RPAREN
                              | COMMA ROUND_ROBIN LPAREN RPAREN
                              | empty"""
        if len(p) > 2:
            if p[2] == "HASH":
                p[0] = p[4]
            elif p[2] in ("BROADCAST", "ROUND_ROBIN"):
                p[0] = p[2]
            else:
                p[0] = p[3]
        else:
            p[0] = None

    @staticmethod
    def p_expression_id(p):
        'expression : unreserved_id'
        p[0] = ('ALIAS', p[1])

    @staticmethod
    def p_sexpr_list(p):
        """sexpr_list : sexpr_list COMMA sexpr
                      | sexpr"""
        if len(p) == 4:
            p[0] = p[1] + (p[3],)
        else:
            p[0] = (p[1],)

    @staticmethod
    def p_expression_table_literal(p):
        'expression : table_literal'
        p[0] = ('TABLE', p[1])

    @staticmethod
    def p_table_literal(p):
        'table_literal : LBRACKET emit_arg_list RBRACKET'
        p[0] = p[2]

    @staticmethod
    def p_expression_empty(p):
        'expression : EMPTY LPAREN column_def_list RPAREN'
        p[0] = ('EMPTY', scheme.Scheme(p[3]))

    @staticmethod
    def p_expression_scan(p):
        'expression : SCAN LPAREN relation_key RPAREN'
        p[0] = ('SCAN', p[3])

    @staticmethod
    def p_expression_samplescan(p):
        """expression : SAMPLESCAN LPAREN relation_key COMMA INTEGER_LITERAL RPAREN
                      | SAMPLESCAN LPAREN relation_key COMMA INTEGER_LITERAL MOD RPAREN
                      | SAMPLESCAN LPAREN relation_key COMMA FLOAT_LITERAL MOD RPAREN
                      | SAMPLESCAN LPAREN relation_key COMMA INTEGER_LITERAL COMMA string_arg RPAREN
                      | SAMPLESCAN LPAREN relation_key COMMA INTEGER_LITERAL MOD COMMA string_arg RPAREN
                      | SAMPLESCAN LPAREN relation_key COMMA FLOAT_LITERAL MOD COMMA string_arg RPAREN"""  # noqa
        if len(p) in (7, 8):
            # Default if no sample type specified
            samp_type = 'WR'
        elif len(p) == 9:
            samp_type = p[7]
        else:
            samp_type = p[8]
        is_pct = p[6] == '%'
        p[0] = ('SAMPLESCAN', p[3], p[5], is_pct, samp_type)

    @staticmethod
    def p_expression_load(p):
        'expression : LOAD LPAREN STRING_LITERAL COMMA file_parser_fun RPAREN'
        format, schema, options = p[5]
        p[0] = ('LOAD', p[3], format, scheme.Scheme(schema), options)

    @staticmethod
    def p_relation_key(p):
        """relation_key : string_arg
                        | string_arg COLON string_arg
                        | string_arg COLON string_arg COLON string_arg"""
        p[0] = relation_key.RelationKey.from_string(''.join(p[1:]))

    # Note: column list cannot be empty
    @staticmethod
    def p_column_def_list(p):
        """column_def_list : column_def_list COMMA column_def
                           | column_def"""
        if len(p) == 4:
            cols = p[1] + [p[3]]
        else:
            cols = [p[1]]
        p[0] = cols

    @staticmethod
    def p_column_def(p):
        'column_def : unreserved_id COLON type_name'
        p[0] = (p[1], p[3])

    @staticmethod
    def p_schema_fun(p):
        'schema_fun : SCHEMA LPAREN column_def_list RPAREN'
        p[0] = p[3]

    @staticmethod
    def p_file_parser_fun(p):
        """file_parser_fun : CSV LPAREN \
   schema_fun COMMA option_list RPAREN
 | CSV LPAREN schema_fun RPAREN
 | OPP LPAREN RPAREN
 | TIPSY LPAREN implicit_tipsy_schema empty option_list RPAREN
 | TIPSY LPAREN implicit_tipsy_schema RPAREN"""
        if len(p) == 7:
            format, schema, options = (p[1], p[3], dict(p[5]))
        elif len(p) == 5:
            format, schema, options = (p[1], p[3], {})
        else:
            format, schema, options = (p[1], [], {})
        p[0] = (format, schema, options)

    @staticmethod
    def p_tipsy_schema(p):
        """implicit_tipsy_schema : empty"""
        p[0] = [("iOrder", "LONG_TYPE"),
                ("mass", "FLOAT_TYPE"),
                ("x", "FLOAT_TYPE"),
                ("y", "FLOAT_TYPE"),
                ("z", "FLOAT_TYPE"),
                ("vx", "FLOAT_TYPE"),
                ("vy", "FLOAT_TYPE"),
                ("vz", "FLOAT_TYPE"),
                ("rho", "FLOAT_TYPE"),
                ("temp", "FLOAT_TYPE"),
                ("hsmooth", "FLOAT_TYPE"),
                ("metals", "FLOAT_TYPE"),
                ("tform", "FLOAT_TYPE"),
                ("eps", "FLOAT_TYPE"),
                ("phi", "FLOAT_TYPE"),
                ("grp", "FLOAT_TYPE"),
                ("type", "FLOAT_TYPE")]

    @staticmethod
    def p_option_list(p):
        """option_list : option_list COMMA option
                           | option"""
        if len(p) == 4:
            opts = p[1] + [p[3]]
        else:
            opts = [p[1]]
        p[0] = opts

    @staticmethod
    def p_option(p):
        'option : unreserved_id EQUALS literal_arg'
        p[0] = (p[1], p[3])

    @staticmethod
    def p_literal_arg(p):
        """literal_arg : STRING_LITERAL
                       | INTEGER_LITERAL
                       | FLOAT_LITERAL
                       | TRUE
                       | FALSE"""
        p[0] = p[1]

    @staticmethod
    def p_type_name(p):
        """type_name : STRING
                     | INT
                     | BOOLEAN
                     | FLOAT"""
        p[0] = myrial_type_map[p[1]]

    @staticmethod
    def p_string_arg(p):
        """string_arg : unreserved_id
                      | STRING_LITERAL"""
        p[0] = p[1]

    @staticmethod
    def p_expression_bagcomp(p):
        'expression : LBRACKET FROM from_arg_list opt_where_clause \
        EMIT emit_arg_list RBRACKET'
        p[0] = ('BAGCOMP', p[3], p[4], p[6])

    @staticmethod
    def p_from_arg_list(p):
        """from_arg_list : from_arg_list COMMA from_arg
                         | from_arg"""
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_from_arg(p):
        """from_arg : expression optional_as unreserved_id
                    | unreserved_id"""
        expr = None
        if len(p) == 4:
            expr = p[1]
            _id = p[3]
        else:
            _id = p[1]
        p[0] = (_id, expr)

    @staticmethod
    def p_optional_as(p):
        """optional_as : AS
                       | empty"""
        p[0] = None

    @staticmethod
    def p_opt_where_clause(p):
        """opt_where_clause : WHERE sexpr
                            | empty"""
        if len(p) == 3:
            p[0] = p[2]
        else:
            p[0] = None

    @staticmethod
    def p_emit_arg_list(p):
        """emit_arg_list : emit_arg_list COMMA emit_arg
                         | emit_arg"""
        if len(p) == 4:
            p[0] = p[1] + (p[3],)
        else:
            p[0] = (p[1],)

    @staticmethod
    def p_emit_arg_explicit(p):
        """emit_arg : sexpr AS LBRACKET unreserved_id_list RBRACKET
                    | sexpr AS unreserved_id
                    | sexpr"""

        sx = p[1]
        names = None
        if len(p) == 6:
            names = p[4]
        if len(p) == 4:
            names = [p[3]]

        emitters = get_emitters(sx)
        if names is not None and len(emitters) != len(names):
            raise IllegalColumnNamesException(p.lineno(0))

        # Verify that there are no nested aggregate expressions
        for ssx in emitters:
            sexpr.check_no_nested_aggregate(ssx, p.lineno(0))

        # Verify that there are no remaining tuple expressions
        for ssx in emitters:
            check_no_tuple_expression(ssx, p.lineno(0))

        p[0] = emitarg.NaryEmitArg(names, emitters, Parser.statemods)
        Parser.statemods = []

    @staticmethod
    def p_emit_arg_table_wildcard(p):
        """emit_arg : unreserved_id DOT TIMES"""
        p[0] = emitarg.TableWildcardEmitArg(p[1])

    @staticmethod
    def p_emit_arg_full_wildcard(p):
        """emit_arg : TIMES"""
        p[0] = emitarg.FullWildcardEmitArg()

    @staticmethod
    def p_expression_select_from_where(p):
        """expression : LPAREN select_from_where RPAREN"""
        p[0] = p[2]

    @staticmethod
    def p_select_from_where(p):
        'select_from_where : SELECT opt_distinct emit_arg_list FROM from_arg_list opt_where_clause opt_limit'  # noqa
        p[0] = ('SELECT', SelectFromWhere(distinct=p[2], select=p[3],
                                          from_=p[5], where=p[6], limit=p[7]))

    @staticmethod
    def p_opt_distinct(p):
        """opt_distinct : DISTINCT
                        | empty"""
        # p[1] is either 'DISTINCT' or None. Use Python truthiness
        p[0] = bool(p[1])

    @staticmethod
    def p_opt_limit(p):
        """opt_limit : LIMIT INTEGER_LITERAL
                     | empty"""
        if len(p) == 3:
            p[0] = p[2]
        else:
            p[0] = None

    @staticmethod
    def p_expression_limit(p):
        'expression : LIMIT LPAREN expression COMMA INTEGER_LITERAL RPAREN'
        p[0] = ('LIMIT', p[3], p[5])

    @staticmethod
    def p_expression_distinct(p):
        'expression : DISTINCT LPAREN expression RPAREN'
        p[0] = ('DISTINCT', p[3])

    @staticmethod
    def p_expression_countall(p):
        'expression : COUNTALL LPAREN expression RPAREN'
        p[0] = ('COUNTALL', p[3])

    @staticmethod
    def p_expression_binary_set_operation(p):
        'expression : setop LPAREN expression COMMA expression RPAREN'
        p[0] = (p[1], p[3], p[5])

    @staticmethod
    def p_expression_unionall(p):
        'expression : UNIONALL LPAREN expression_list RPAREN'
        p[0] = ('UNIONALL', p[3])

    @staticmethod
    def p_expression_list(p):
        """expression_list : expression COMMA expression_list
                           | expression"""
        if len(p) == 4:
            p[0] = [p[1]] + p[3]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_setop(p):
        """setop : INTERSECT
                 | DIFF
                 | UNION"""
        p[0] = p[1]

    @staticmethod
    def p_expression_unionall_plus_inline(p):
        """expression : expression PLUS expression"""
        p[0] = ('UNIONALL', [p[1], p[3]])

    @staticmethod
    def p_expression_cross(p):
        'expression : CROSS LPAREN expression COMMA expression RPAREN'
        p[0] = ('CROSS', p[3], p[5])

    @staticmethod
    def p_expression_join(p):
        'expression : JOIN LPAREN join_argument COMMA join_argument RPAREN'
        if len(p[3].columns) != len(p[5].columns):
            raise JoinColumnCountMismatchException()
        p[0] = ('JOIN', p[3], p[5])

    @staticmethod
    def p_join_argument_list(p):
        'join_argument : expression COMMA LPAREN column_ref_list RPAREN'
        p[0] = JoinTarget(p[1], p[4])

    @staticmethod
    def p_join_argument_single(p):
        'join_argument : expression COMMA column_ref'
        p[0] = JoinTarget(p[1], [p[3]])

    # column_ref refers to the name or position of a column; these serve
    # as arguments to join.
    @staticmethod
    def p_column_ref_list(p):
        """column_ref_list : column_ref_list COMMA column_ref
                           | column_ref"""
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_column_ref_string(p):
        'column_ref : unreserved_id'
        p[0] = p[1]

    @staticmethod
    def p_column_ref_index(p):
        'column_ref : DOLLAR INTEGER_LITERAL'
        p[0] = p[2]

    # scalar expressions map to raco.Expression instances; these are operations
    # that return scalar types.

    @staticmethod
    def p_sexpr_integer_literal(p):
        'sexpr : INTEGER_LITERAL'
        p[0] = sexpr.NumericLiteral(p[1])

    @staticmethod
    def p_sexpr_string_literal(p):
        'sexpr : STRING_LITERAL'
        p[0] = sexpr.StringLiteral(p[1])

    @staticmethod
    def p_sexpr_float_literal(p):
        'sexpr : FLOAT_LITERAL'
        p[0] = sexpr.NumericLiteral(p[1])

    @staticmethod
    def p_sexpr_boolean_literal(p):
        '''sexpr : TRUE
                 | FALSE'''
        bv = p[1] == 'TRUE'
        p[0] = sexpr.BooleanLiteral(bv)

    @staticmethod
    def p_sexpr_id(p):
        'sexpr : unreserved_id'
        try:
            # Check for zero-argument function
            p[0] = Parser.resolve_function(p, p[1], [])
        except:
            # Resolve as an attribute reference
            p[0] = sexpr.NamedAttributeRef(p[1])

    @staticmethod
    def p_sexpr_index(p):
        'sexpr : DOLLAR INTEGER_LITERAL'
        p[0] = sexpr.UnnamedAttributeRef(p[2])

    @staticmethod
    def p_sexpr_id_dot_ref(p):
        'sexpr : unreserved_id DOT column_ref'
        p[0] = sexpr.DottedRef(p[1], p[3])

    @staticmethod
    def p_sexpr_group(p):
        'sexpr : LPAREN sexpr RPAREN'
        p[0] = p[2]

    @staticmethod
    def p_sexpr_uminus(p):
        'sexpr : MINUS sexpr %prec UMINUS'
        p[0] = sexpr.TIMES(sexpr.NumericLiteral(-1), p[2])

    @staticmethod
    def p_sexpr_worker_id(p):
        """sexpr : WORKER_ID LPAREN RPAREN"""
        p[0] = sexpr.WORKERID()

    @staticmethod
    def p_sexpr_binop(p):
        """sexpr : sexpr PLUS sexpr
                   | sexpr MINUS sexpr
                   | sexpr TIMES sexpr
                   | sexpr DIVIDE sexpr
                   | sexpr IDIVIDE sexpr
                   | sexpr MOD sexpr
                   | sexpr GT sexpr
                   | sexpr LT sexpr
                   | sexpr GE sexpr
                   | sexpr GE2 sexpr
                   | sexpr LE sexpr
                   | sexpr LE2 sexpr
                   | sexpr NE sexpr
                   | sexpr NE2 sexpr
                   | sexpr NE3 sexpr
                   | sexpr EQ sexpr
                   | sexpr EQUALS sexpr
                   | sexpr AND sexpr
                   | sexpr OR sexpr
                   | sexpr LIKE sexpr"""
        p[0] = binops[p[2]](p[1], p[3])

    @staticmethod
    def p_sexpr_not(p):
        'sexpr : NOT sexpr'
        p[0] = sexpr.NOT(p[2])

    @staticmethod
    def resolve_stateful_func(func, args):
        """Resolve a stateful function given argument expressions.

        :param func: An instance of StatefulFunc
        :param args: A list of argument expressions
        :return: An emit expression and a StateVar list.  All expressions
        have no free variables.
        """
        assert isinstance(func, StatefulFunc)
        state_var_names = func.statemods.keys()

        # Mangle state variable names to allow multiple invocations to coexist
        state_vars_mangled = [Parser.mangle(sv) for sv in state_var_names]
        mangle_dict = dict(zip(state_var_names, state_vars_mangled))

        statemods = []
        for name, (init_expr, update_expr) in func.statemods.iteritems():
            # Convert state mod references into appropriate expressions
            update_expr = sexpr.resolve_state_vars(update_expr,  # noqa
                state_var_names, mangle_dict)
            # Convert argument references into appropriate expressions
            update_expr = sexpr.resolve_function(update_expr,  # noqa
                dict(zip(func.args, args)))
            statemods.append(StateVar(mangle_dict[name],
                                      init_expr, update_expr))
        emit_expr = sexpr.resolve_state_vars(func.sexpr, state_var_names,
                                             mangle_dict)
        return emit_expr, statemods

    @staticmethod
    def resolve_function(p, name, args):
        """Resolve a function invocation into an Expression instance.

        :param p: The parser context
        :param name: The name of the function
        :type name: string
        :param args: A list of argument expressions
        :type args: list of raco.expression.Expression instances
        :return: An expression with no free variables.
        """

        # try to get function from udf or system defined functions
        if name in Parser.udf_functions:
            func = Parser.udf_functions[name]
        else:
            func = expr_lib.lookup(name, len(args))

        if isinstance(func, VariadicFunction):
            func = func.bind(*args)

        if func is None:
            raise NoSuchFunctionException(name, p.lineno(0))
        if len(func.args) != len(args):
            raise InvalidArgumentList(name, func.args, p.lineno(0))

        if isinstance(func, Function):
            return sexpr.resolve_function(func.sexpr, dict(zip(func.args, args)))  # noqa
        elif isinstance(func, StatefulFunc):
            emit_expr, statemods = Parser.resolve_stateful_func(func, args)
            Parser.statemods.extend(statemods)

            # If the aggregate is decomposable, construct local and remote
            # emitters and statemods.
            if name in Parser.decomposable_aggs:
                ds = Parser.decomposable_aggs[name]
                local_emit, local_statemods = Parser.resolve_stateful_func(
                    ds.local, args)

                # Problem: we must connect the local aggregate outputs to
                # the remote aggregate inputs.  At this stage, we don't have
                # enough information to construct argument expressions to
                # serve as input to the remote aggregate.  Instead, we
                # introduce a placeholder reference, which is referenced
                # relative to the start of the local aggregate output.
                remote_args = [sexpr.LocalAggregateOutput(i)
                               for i in range(len(ds.remote.args))]
                remote_emit, remote_statemods = Parser.resolve_stateful_func(
                    ds.remote, remote_args)

                # local and remote emitters may be tuple-valued; flatten them.
                local_emitters = get_emitters(local_emit)
                remote_emitters = get_emitters(remote_emit)
                ds = sexpr.DecomposableAggregateState(
                    local_emitters, local_statemods,
                    remote_emitters, remote_statemods)

                # Associate a decomposable state structure with the first
                # emitter.  Mark the remaining emitters as decomposable, but
                # without their own associated decomposed emitters and
                # statemods.
                emitters = get_emitters(emit_expr)
                emitters[0].set_decomposable_state(ds)
                for emt in emitters[1:]:
                    emt.set_decomposable_state(
                        sexpr.DecomposableAggregateState())
            return emit_expr
        else:
            assert False

    @staticmethod
    def p_sexpr_function_k_args(p):
        'sexpr : ID LPAREN function_param_list RPAREN'
        p[0] = Parser.resolve_function(p, p[1], p[3])

    @staticmethod
    def p_sexpr_function_zero_args(p):
        'sexpr : ID LPAREN RPAREN'
        p[0] = Parser.resolve_function(p, p[1], [])

    @staticmethod
    def p_function_param_list(p):
        """function_param_list : function_param_list COMMA sexpr
                               | sexpr"""
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_sexpr_countall(p):
        'sexpr : COUNTALL LPAREN RPAREN'
        p[0] = Parser.resolve_function(p, 'COUNTALL', [])

    @staticmethod
    def p_sexpr_count(p):
        'sexpr : COUNT LPAREN count_arg RPAREN'
        if p[3] == '*':
            p[0] = sexpr.COUNTALL()
        else:
            p[0] = sexpr.COUNT(p[3])

    @staticmethod
    def p_sexpr_cast(p):
        """sexpr : type_name LPAREN sexpr RPAREN"""
        p[0] = sexpr.CAST(p[1], p[3])

    @staticmethod
    def p_count_arg(p):
        """count_arg : TIMES
                     | sexpr"""
        p[0] = p[1]

    @staticmethod
    def p_sexpr_unbox(p):
        'sexpr : TIMES unreserved_id optional_column_ref'
        p[0] = sexpr.Unbox(p[2], p[3])

    @staticmethod
    def p_when_expr(p):
        'when_expr : WHEN sexpr THEN sexpr'
        p[0] = (p[2], p[4])

    @staticmethod
    def p_when_expr_list(p):
        """when_expr_list : when_expr_list when_expr
                          | when_expr
        """
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_sexpr_case(p):
        'sexpr : CASE when_expr_list ELSE sexpr END'
        p[0] = sexpr.Case(p[2], p[4])

    @staticmethod
    def p_optional_column_ref(p):
        """optional_column_ref : DOT column_ref
                               | empty"""
        if len(p) == 3:
            p[0] = p[2]
        else:
            p[0] = None

    @staticmethod
    def p_empty(p):
        'empty :'
        pass

    def parse(self, s, udas=None):
        scanner.lexer.lineno = 1
        Parser.udf_functions = {}
        Parser.decomposable_aggs = {}
        map(lambda uda: self.add_python_udf(*uda), udas or [])
        parser = yacc.yacc(module=self, debug=False, optimize=False)
        stmts = parser.parse(s, lexer=scanner.lexer, tracking=True)

        # Strip out the remnants of parsed functions to leave only a list of
        # statements
        return [st for st in stmts if st is not None]

    @staticmethod
    def p_error(token):
        if token:
            raise MyrialParseException(token)
        else:
            raise MyrialUnexpectedEndOfFileException()
