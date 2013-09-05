#!/usr/bin/env python

import ply.yacc as yacc

import raco.myrial.scanner as scanner
import raco.scheme as scheme
import raco.expression as colexpr

import collections
import sys

class JoinColumnCountMismatchException(Exception):
    pass

class ParseException(Exception):
    pass

# ID is a symbol name that identifies an input relation; columns is a list of
# "atomic" column expressions -- either NamedAttributeRef (for strings) or
# UnnamedAttributeRef (for positions).
JoinTarget = collections.namedtuple('JoinTarget',['id', 'columns'])

class RelationKey(object):
    def __init__(self, table, program='default', user='nobody'):
        self.table = table
        self.program = program
        self.user = user

    def __repr__(self):
        return 'RelationKey(%s,%s,%s)' % (self.table, self.program,self.user)

# Mapping from source symbols to raco.expression.BinaryOperator classes
binops = {
    '+': colexpr.PLUS,
    '-' : colexpr.MINUS,
    '/' : colexpr.DIVIDE,
    '*' : colexpr.TIMES,
    '>' : colexpr.GT,
    '<' : colexpr.LT,
    '>=' : colexpr.GTEQ,
    '<=' : colexpr.LTEQ,
    '!=' : colexpr.NEQ,
    '==' : colexpr.EQ,
    '&&' : colexpr.AND,
    '||' : colexpr.OR,
}

class Parser:
    # Precedence among column expression operators in ascending order; this is
    # necessary to disambiguate the grammer.  Operator precedence is identical
    # to C.  http://en.cppreference.com/w/cpp/language/operator_precedence

    precedence = (
        ('left', 'LOR'),
        ('left', 'LAND'),
        ('left', 'EQ'),
        ('left', 'NE'),
        ('left', 'GT', 'LT', 'LE', 'GE'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE'),
        ('right', 'LNOT'),
        ('right', 'UMINUS'), # Unary minus operator (for negative numbers)
    )

    def __init__(self, log=yacc.PlyLogger(sys.stderr)):
        self.log = log
        self.tokens = scanner.tokens

    def p_statement_list(self, p):
        '''statement_list : statement_list statement
                          | statement'''
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[0] = [p[1]]

    def p_statement_assign(self, p):
        'statement : ID EQUALS expression SEMI'
        p[0] = ('ASSIGN', p[1], p[3])

    def p_statement_dump(self, p):
        'statement : DUMP ID SEMI'
        p[0] = ('DUMP', p[2])

    def p_statement_describe(self, p):
        'statement : DESCRIBE ID SEMI'
        p[0] = ('DESCRIBE', p[2])

    def p_statement_explain(self, p):
        'statement : EXPLAIN ID SEMI'
        p[0] = ('EXPLAIN', p[2])

    def p_statement_dowhile(self, p):
        'statement : DO statement_list WHILE expression SEMI'
        p[0] = ('DOWHILE', p[2], p[4])

    def p_statement_store(self, p):
        'statement : STORE ID COMMA relation_key SEMI'
        p[0] = ('STORE', p[2], p[4])

    def p_expression_id(self, p):
        'expression : ID'
        p[0] = ('ALIAS', p[1])

    def p_expression_scan(self, p):
        'expression : SCAN relation_key optional_as'
        # TODO(AJW): Nix optional schema argument once we can read this from
        # myrial?
        p[0] = ('SCAN', p[2], p[3])

    def p_relation_key(self, p):
        'relation_key : LBRACE string_arg_list RBRACE'
        # {table [, program] [,user]}
        if len(p[2]) < 1:
            raise ParseException("No table name provided")
        if len(p[2]) > 3:
            raise ParseException("Too many arguments to relation key")
        p[0] = RelationKey(*p[2])

    def p_optional_as(self, p):
        '''optional_as : AS schema
                       | empty'''
        if len(p) == 3:
            p[0] = p[2]
        else:
            p[0] = None

    def p_schema(self, p):
        'schema : LPAREN column_def_list RPAREN'
        p[0] = scheme.Scheme(p[2])

    def p_column_def_list(self, p):
        '''column_def_list : column_def_list COMMA column_def
                           | column_def'''
        if len(p) == 4:
            cols = p[1] + [p[3]]
        else:
            cols = [p[1]]
        p[0] = cols

    def p_column_def(self, p):
        'column_def : ID COLON type_name'
        p[0] = (p[1], p[3])

    def p_type_name(self, p):
        '''type_name : STRING
                     | INT'''
        p[0] = p[1]

    # For operators that take string-like arguments: allow unquoted
    # identifiers and quoted strings to be used equivalently
    def p_string_arg_list(self, p):
        '''string_arg_list : string_arg_list COMMA string_arg
                           | string_arg'''
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    def p_string_arg(self, p):
        '''string_arg : ID
                      | STRING_LITERAL'''
        p[0] = p[1]

    def p_expression_limit(self, p):
        'expression : LIMIT ID COMMA INTEGER_LITERAL'
        p[0] = ('LIMIT', p[2], p[4])

    def p_expression_distinct(self, p):
        'expression : DISTINCT ID'
        p[0] = ('DISTINCT', p[2])

    def p_expression_count(self, p):
        'expression : COUNT ID'
        p[0] = ('COUNT', p[2])

    def p_expression_binary_set_operation(self, p):
        'expression : setop ID COMMA ID'
        p[0] = (p[1], p[2], p[4])

    def p_setop(self, p):
        '''setop : INTERSECT
                 | DIFF
                 | UNIONALL'''
        p[0] = p[1]

    def p_expression_join(self, p):
        'expression : JOIN join_argument COMMA join_argument'
        if len(p[2].columns) != len(p[4].columns):
            raise JoinColumnCountMismatchException()
        p[0] = ('JOIN', p[2], p[4])

    def p_join_argument_list(self, p):
        'join_argument : ID BY LPAREN column_ref_list RPAREN'
        p[0] = JoinTarget(p[1], p[4])

    def p_join_argument_single(self, p):
        'join_argument : ID BY column_ref'
        p[0] = JoinTarget(p[1], [p[3]])

    # column_ref refers to the name or position of a column; these serve
    # as arguments to join.
    def p_column_ref_list(self, p):
        '''column_ref_list : column_ref_list COMMA column_ref
                           | column_ref'''
        if len(p) == 4:
            cols = p[1] + [p[3]]
        else:
            cols = [p[1]]
        p[0] = cols

    def p_column_ref_id(self, p):
        'column_ref : ID'
        p[0] = colexpr.NamedAttributeRef(p[1])

    def p_column_ref_index(self, p):
        'column_ref : DOLLAR INTEGER_LITERAL'
        p[0] = colexpr.UnnamedAttributeRef(p[2])

    def p_apply_expr(self, p):
        'expression : APPLY ID EMIT LPAREN apply_arg_list RPAREN'
        p[0] = ('APPLY', p[2], dict(p[5]))

    def p_apply_arg_list(self, p):
        '''apply_arg_list : apply_arg_list COMMA apply_arg
                          | apply_arg'''
        # Resolves into a list of tuples of the form (id, raco.Expression)
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    def p_apply_arg(self, p):
        'apply_arg : ID EQUALS colexpr'
        p[0] = (p[1], p[3])

    # column expressions map to raco.Expression instances; these are operations
    # that return atomic types, and that are suitable as arguments for apply.

    def p_colexpr_integer_literal(self, p):
        'colexpr : INTEGER_LITERAL'
        p[0] = colexpr.NumericLiteral(p[1])

    def p_colexpr_string_literal(self, p):
        'colexpr : STRING_LITERAL'
        p[0] = colexpr.StringLiteral(p[1])

    def p_colexpr_id(self, p):
        'colexpr : ID'
        p[0] = colexpr.NamedAttributeRef(p[1])

    def p_colexpr_index(self, p):
        'colexpr : DOLLAR INTEGER_LITERAL'
        p[0] = colexpr.UnnamedAttributeRef(p[2])

    def p_colexpr_group(self, p):
        'colexpr : LPAREN colexpr RPAREN'
        p[0] = p[2]

    def p_colexpr_uminus(self, p):
        'colexpr : MINUS colexpr %prec UMINUS'
        p[0] = colexpr.TIMES(colexpr.NumericLiteral(-1), t[2])

    def p_expression_binop(self, p):
        '''colexpr : colexpr binary_op colexpr'''
        p[0] = binops[p[2]](p[1], p[3])

    def p_binary_op(self, p):
        '''binary_op : PLUS
                     | MINUS
                     | TIMES
                     | DIVIDE
                     | GT
                     | LT
                     | GE
                     | LE
                     | EQ'''
        p[0] = p[1]

    def p_colexpr_not(self, p):
        'colexpr : LNOT colexpr'
        p[0] = colexpr.NOT(p[2])

    def p_empty(self, p):
        'empty :'
        pass

    def parse(self, s):
        parser = yacc.yacc(module=self, debug=True)
        return parser.parse(s, lexer=scanner.lexer, tracking=True)

    def p_error(self, p):
        self.log.error("Syntax error: %s" %  str(p))
