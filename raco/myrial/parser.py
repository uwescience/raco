import collections
import sys

from ply import yacc

from raco import relation_key
import raco.myrial.scanner as scanner
import raco.scheme as scheme
import raco.expression as sexpr
import raco.myrial.emitarg as emitarg
from .exceptions import *

class JoinColumnCountMismatchException(Exception):
    pass

# ID is a symbol name that identifies an input expression; columns is a list of
# columns expressed as either names or integer positions.
JoinTarget = collections.namedtuple('JoinTarget', ['expr', 'columns'])

SelectFromWhere = collections.namedtuple(
    'SelectFromWhere', ['distinct', 'select','from_', 'where', 'limit'])

# Mapping from source symbols to raco.expression.BinaryOperator classes
binops = {
    '+': sexpr.PLUS,
    '-' : sexpr.MINUS,
    '/' : sexpr.DIVIDE,
    '*' : sexpr.TIMES,
    '>' : sexpr.GT,
    '<' : sexpr.LT,
    '>=' : sexpr.GTEQ,
    '<=' : sexpr.LTEQ,
    '!=' : sexpr.NEQ,
    '<>' : sexpr.NEQ,
    '==' : sexpr.EQ,
    '=' : sexpr.EQ,
    'AND' : sexpr.AND,
    'OR' : sexpr.OR,
    'POW' : sexpr.POW,
}

# Mapping from source symbols to raco.expression.UnaryOperator classes
unops = {
    'ABS' : sexpr.ABS,
    'CEIL' : sexpr.CEIL,
    'COS' : sexpr.COS,
    'FLOOR' : sexpr.FLOOR,
    'LOG' : sexpr.LOG,
    'SIN' : sexpr.SIN,
    'SQRT' : sexpr.SQRT,
    'TAN' : sexpr.TAN,
}

class Parser(object):
    def __init__(self, log=yacc.PlyLogger(sys.stderr)):
        self.log = log
        self.tokens = scanner.tokens

        # Precedence among scalar expression operators in ascending order; this
        # is necessary to disambiguate the grammer.  Operator precedence is
        # identical to Python:
        # http://docs.python.org/2/reference/expressions.html#comparisons

        self.precedence = (
            ('left', 'OR'),
            ('left', 'AND'),
            ('right', 'NOT'),
            ('left', 'EQ', 'NE', 'GT', 'LT', 'LE', 'GE'),
            ('left', 'PLUS', 'MINUS'),
            ('left', 'TIMES', 'DIVIDE'),
            ('right', 'UMINUS'), # Unary minus operator (for negative numbers)
        )

    @staticmethod
    def p_statement_list(p):
        '''statement_list : statement_list statement
                          | statement'''
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_statement_assign(p):
        'statement : ID EQUALS rvalue SEMI'
        p[0] = ('ASSIGN', p[1], p[3])

    # expressions must be embeddable in other expressions; certain constructs
    # are not embeddable, but are available as r-values in an assignment 
    @staticmethod
    def p_rvalue(p):
        """rvalue : expression
                  | select_from_where"""
        p[0] = p[1]

    @staticmethod
    def p_statement_dump(p):
        'statement : DUMP LPAREN ID RPAREN SEMI'
        p[0] = ('DUMP', p[3])

    @staticmethod
    def p_statement_describe(p):
        'statement : DESCRIBE LPAREN ID RPAREN SEMI'
        p[0] = ('DESCRIBE', p[3])

    @staticmethod
    def p_statement_explain(p):
        'statement : EXPLAIN LPAREN ID RPAREN SEMI'
        p[0] = ('EXPLAIN', p[3])

    @staticmethod
    def p_statement_dowhile(p):
        'statement : DO statement_list WHILE expression SEMI'
        p[0] = ('DOWHILE', p[2], p[4])

    @staticmethod
    def p_statement_store(p):
        'statement : STORE LPAREN ID COMMA relation_key RPAREN SEMI'
        p[0] = ('STORE', p[3], p[5])

    @staticmethod
    def p_expression_id(p):
        'expression : ID'
        p[0] = ('ALIAS', p[1])

    @staticmethod
    def p_expression_table_literal(p):
        'expression : LBRACKET emit_arg_list RBRACKET'
        p[0] = ('TABLE', p[2])

    @staticmethod
    def p_expression_empty(p):
        'expression : EMPTY LPAREN optional_schema RPAREN'
        p[0] = ('EMPTY', p[3])

    @staticmethod
    def p_expression_scan(p):
        'expression : SCAN LPAREN relation_key RPAREN'
        p[0] = ('SCAN', p[3])

    @staticmethod
    def p_relation_key(p):
        '''relation_key : string_arg
                        | string_arg COLON string_arg
                        | string_arg COLON string_arg COLON string_arg'''
        p[0] = relation_key.RelationKey.from_string(''.join(p[1:]))

    @staticmethod
    def p_optional_schema(p):
        '''optional_schema : column_def_list
                           | empty'''
        if len(p) == 2:
            p[0] = scheme.Scheme(p[1])
        else:
            p[0] = None

    # Note: column list cannot be empty
    @staticmethod
    def p_column_def_list(p):
        '''column_def_list : column_def_list COMMA column_def
                           | column_def'''
        if len(p) == 4:
            cols = p[1] + [p[3]]
        else:
            cols = [p[1]]
        p[0] = cols

    @staticmethod
    def p_column_def(p):
        'column_def : ID COLON type_name'
        p[0] = (p[1], p[3])

    @staticmethod
    def p_type_name(p):
        '''type_name : STRING
                     | INT
                     | FLOAT'''
        p[0] = p[1]

    @staticmethod
    def p_string_arg(p):
        '''string_arg : ID
                      | STRING_LITERAL'''
        p[0] = p[1]

    @staticmethod
    def p_expression_bagcomp(p):
        'expression : LBRACKET FROM from_arg_list opt_where_clause \
        EMIT emit_arg_list RBRACKET'
        p[0] = ('BAGCOMP', p[3], p[4], p[6])

    @staticmethod
    def p_from_arg_list(p):
        '''from_arg_list : from_arg_list COMMA from_arg
                         | from_arg'''
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_from_arg(p):
        '''from_arg : expression optional_as ID
                    | ID'''
        expr = None
        if len(p) == 4:
            expr = p[1]
            _id = p[3]
        else:
            _id = p[1]
        p[0] = (_id, expr)

    @staticmethod
    def p_optional_as(p):
        '''optional_as : AS
                       | empty'''
        p[0] = None

    @staticmethod
    def p_opt_where_clause(p):
        '''opt_where_clause : WHERE sexpr
                            | empty'''
        if len(p) == 3:
            p[0] = p[2]
        else:
            p[0] = None

    @staticmethod
    def p_emit_arg_list(p):
        '''emit_arg_list : emit_arg_list COMMA emit_arg
                         | emit_arg'''
        if len(p) == 4:
            p[0] = p[1] + (p[3],)
        else:
            p[0] = (p[1],)

    @staticmethod
    def p_emit_arg_singleton(p):
        '''emit_arg : sexpr AS ID
                    | sexpr'''
        if len(p) == 4:
            name = p[3]
            sexpr = p[1]
        else:
            name = None
            sexpr = p[1]
        p[0] = emitarg.SingletonEmitArg(name, sexpr)

    @staticmethod
    def p_emit_arg_table_wildcard(p):
        '''emit_arg : ID DOT TIMES'''
        p[0] = emitarg.TableWildcardEmitArg(p[1])

    @staticmethod
    def p_emit_arg_full_wildcard(p):
        '''emit_arg : TIMES'''
        p[0] = emitarg.FullWildcardEmitArg()

    @staticmethod
    def p_expression_select_from_where(p):
        """expression : LPAREN select_from_where RPAREN"""
        p[0] = p[2]

    @staticmethod
    def p_select_from_where(p):
        'select_from_where : SELECT opt_distinct emit_arg_list FROM from_arg_list opt_where_clause opt_limit'
        p[0] = ('SELECT', SelectFromWhere(distinct=p[2], select=p[3],
                                          from_=p[5], where=p[6], limit=p[7]))

    @staticmethod
    def p_opt_distinct(p):
        '''opt_distinct : DISTINCT
                        | empty'''
        # p[1] is either 'DISTINCT' or None. Use Python truthiness
        p[0] = bool(p[1])

    @staticmethod
    def p_opt_limit(p):
        '''opt_limit : LIMIT INTEGER_LITERAL
                     | empty'''
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
    def p_setop(p):
        '''setop : INTERSECT
                 | DIFF
                 | UNIONALL'''
        p[0] = p[1]

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
        '''column_ref_list : column_ref_list COMMA column_ref
                           | column_ref'''
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    @staticmethod
    def p_column_ref_string(p):
        'column_ref : ID'
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
    def p_sexpr_id(p):
        'sexpr : ID'
        p[0] = sexpr.NamedAttributeRef(p[1])

    @staticmethod
    def p_sexpr_index(p):
        'sexpr : DOLLAR INTEGER_LITERAL'
        p[0] = sexpr.UnnamedAttributeRef(p[2])

    @staticmethod
    def p_sexpr_id_dot_id(p):
        'sexpr : ID DOT ID'
        p[0] = sexpr.Unbox(p[1], p[3])

    @staticmethod
    def p_sexpr_id_dot_pos(p):
        'sexpr : ID DOT DOLLAR INTEGER_LITERAL'
        p[0] = sexpr.Unbox(p[1], p[4])

    @staticmethod
    def p_sexpr_group(p):
        'sexpr : LPAREN sexpr RPAREN'
        p[0] = p[2]

    @staticmethod
    def p_sexpr_uminus(p):
        'sexpr : MINUS sexpr %prec UMINUS'
        p[0] = sexpr.TIMES(sexpr.NumericLiteral(-1), p[2])

    @staticmethod
    def p_sexpr_unop(p):
        '''sexpr : ABS LPAREN sexpr RPAREN
                   | CEIL LPAREN sexpr RPAREN
                   | COS LPAREN sexpr RPAREN
                   | FLOOR LPAREN sexpr RPAREN
                   | LOG LPAREN sexpr RPAREN
                   | SIN LPAREN sexpr RPAREN
                   | SQRT LPAREN sexpr RPAREN
                   | TAN LPAREN sexpr RPAREN'''
        p[0] = unops[p[1]](p[3])

    @staticmethod
    def p_sexpr_binop(p):
        '''sexpr : sexpr PLUS sexpr
                   | sexpr MINUS sexpr
                   | sexpr TIMES sexpr
                   | sexpr DIVIDE sexpr
                   | sexpr GT sexpr
                   | sexpr LT sexpr
                   | sexpr GE sexpr
                   | sexpr LE sexpr
                   | sexpr NE sexpr
                   | sexpr NE2 sexpr
                   | sexpr EQ sexpr
                   | sexpr EQUALS sexpr
                   | sexpr AND sexpr
                   | sexpr OR sexpr'''
        p[0] = binops[p[2]](p[1], p[3])

    @staticmethod
    def p_sexpr_pow(p):
        'sexpr : POW LPAREN sexpr COMMA sexpr RPAREN'
        p[0] = sexpr.POW(p[3], p[5])

    @staticmethod
    def p_sexpr_not(p):
        'sexpr : NOT sexpr'
        p[0] = sexpr.NOT(p[2])

    @staticmethod
    def p_sexpr_countall(p):
        'sexpr : COUNTALL LPAREN RPAREN'
        p[0] = sexpr.COUNTALL()

    @staticmethod
    def p_sexpr_count(p):
        'sexpr : COUNT LPAREN count_arg RPAREN'
        if p[3] == '*':
            p[0] = sexpr.COUNTALL()
        else:
            p[0] = sexpr.COUNT(p[3])

    @staticmethod
    def p_count_arg(p):
        '''count_arg : TIMES
                     | sexpr'''
        p[0] = p[1]

    @staticmethod
    def p_sexpr_unary_aggregate(p):
        'sexpr : unary_aggregate_func LPAREN sexpr RPAREN'
        p[0] = p[1](p[3])

    @staticmethod
    def p_unary_aggregate_func(p):
        '''unary_aggregate_func : MAX
                                | MIN
                                | SUM
                                | AVG
                                | STDEV'''

        if p[1] == 'MAX': func = sexpr.MAX
        if p[1] == 'MIN': func = sexpr.MIN
        if p[1] == 'SUM': func = sexpr.SUM
        if p[1] == 'AVG': func = sexpr.AVERAGE
        if p[1] == 'STDEV': func = sexpr.STDEV

        p[0] = func

    @staticmethod
    def p_sexpr_unbox(p):
        'sexpr : TIMES expression optional_column_ref'
        p[0] = sexpr.Unbox(p[2], p[3])

    @staticmethod
    def p_optional_column_ref(p):
        '''optional_column_ref : DOT column_ref
                               | empty'''
        if len(p) == 3:
            p[0] = p[2]
        else:
            p[0] = None

    @staticmethod
    def p_empty(p):
        'empty :'
        pass

    def parse(self, s):
        parser = yacc.yacc(module=self, debug=False, optimize=False)
        return parser.parse(s, lexer=scanner.lexer, tracking=True)

    @staticmethod
    def p_error(token):
        if token:
            raise MyrialUnexpectedTokenException(token)
        else:
            raise MyrialUnexpectedEndOfFileException()
