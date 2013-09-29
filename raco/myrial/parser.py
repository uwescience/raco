#!/usr/bin/env python

import ply.yacc as yacc

import raco.myrial.scanner as scanner
import raco.scheme as scheme
import raco.expression as sexpr

import collections
import sys

class JoinColumnCountMismatchException(Exception):
    pass

class ParseException(Exception):
    pass

# ID is a symbol name that identifies an input expression; columns is a list of
# columns expressed as either names or integer positions.
JoinTarget = collections.namedtuple('JoinTarget',['expr', 'columns'])

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
    '==' : sexpr.EQ,
    'AND' : sexpr.AND,
    'OR' : sexpr.OR,
}

class Parser:
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
        'statement : DUMP LPAREN ID RPAREN SEMI'
        p[0] = ('DUMP', p[3])

    def p_statement_describe(self, p):
        'statement : DESCRIBE LPAREN ID RPAREN SEMI'
        p[0] = ('DESCRIBE', p[3])

    def p_statement_explain(self, p):
        'statement : EXPLAIN LPAREN ID RPAREN SEMI'
        p[0] = ('EXPLAIN', p[3])

    def p_statement_dowhile(self, p):
        'statement : DO statement_list WHILE expression SEMI'
        p[0] = ('DOWHILE', p[2], p[4])

    def p_statement_store(self, p):
        'statement : STORE LPAREN ID COMMA relation_key RPAREN SEMI'
        p[0] = ('STORE', p[3], p[5])

    def p_expression_id(self, p):
        'expression : ID'
        p[0] = ('ALIAS', p[1])

    def p_expression_table_literal(self, p):
        'expression : LBRACKET emit_arg_list RBRACKET'
        p[0] = ('TABLE', p[2])

    def p_expression_scan(self, p):
        'expression : SCAN LPAREN relation_key optional_schema RPAREN'
        # TODO(AJW): Nix optional schema argument once we can read this from
        # myrial?
        p[0] = ('SCAN', p[3], p[4])

    def p_relation_key(self, p):
        '''relation_key : string_arg
                        | string_arg COLON string_arg
                        | string_arg COLON string_arg COLON string_arg'''
        p[0] = ''.join(p[1:])

    def p_optional_schema(self, p):
        '''optional_schema : COMMA column_def_list
                           | empty'''
        if len(p) == 3:
            p[0] = scheme.Scheme(p[2])
        else:
            p[0] = None

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

    def p_string_arg(self, p):
        '''string_arg : ID
                      | STRING_LITERAL'''
        p[0] = p[1]

    def p_expression_bagcomp(self, p):
        'expression : LBRACKET FROM from_arg_list opt_where_clause \
        emit_clause RBRACKET'
        p[0] = ('BAGCOMP', p[3], p[4], p[5])

    def p_from_arg_list(self, p):
        '''from_arg_list : from_arg_list COMMA from_arg
                         | from_arg'''
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    def p_from_arg(self, p):
        '''from_arg : ID EQUALS expression
                    | ID'''
        expr = None
        if len(p) == 4:
            expr = p[3]
        p[0] = (p[1], expr)

    def p_opt_where_clause(self, p):
        '''opt_where_clause : WHERE sexpr
                            | empty'''
        if len(p) == 3:
            p[0] = p[2]
        else:
            p[0] = None

    def p_emit_clause_star(self, p):
        '''emit_clause : EMIT TIMES'''
        p[0] = None

    def p_emit_clause_list(self, p):
        '''emit_clause : EMIT emit_arg_list'''
        p[0] = p[2]

    def p_emit_arg_list(self, p):
        '''emit_arg_list : emit_arg_list COMMA emit_arg
                         | emit_arg'''
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    def p_emit_arg(self, p):
        '''emit_arg : string_arg EQUALS sexpr
                    | sexpr'''
        if len(p) == 4:
            p[0] = (p[1], p[3])
        else:
            p[0] = (None, p[1])

    def p_expression_limit(self, p):
        'expression : LIMIT LPAREN expression COMMA INTEGER_LITERAL RPAREN'
        p[0] = ('LIMIT', p[3], p[5])

    def p_expression_distinct(self, p):
        'expression : DISTINCT LPAREN expression RPAREN'
        p[0] = ('DISTINCT', p[3])

    def p_expression_countall(self, p):
        'expression : COUNTALL LPAREN expression RPAREN'
        p[0] = ('COUNTALL', p[3])

    def p_expression_binary_set_operation(self, p):
        'expression : setop ID COMMA ID'
        p[0] = (p[1], p[2], p[4])

    def p_setop(self, p):
        '''setop : INTERSECT
                 | DIFF
                 | UNIONALL'''
        p[0] = p[1]

    def p_expression_cross(self, p):
        'expression : CROSS LPAREN expression COMMA expression RPAREN'
        p[0] = ('CROSS', p[3], p[5])

    def p_expression_join(self, p):
        'expression : JOIN LPAREN join_argument COMMA join_argument RPAREN'
        if len(p[3].columns) != len(p[5].columns):
            raise JoinColumnCountMismatchException()
        p[0] = ('JOIN', p[3], p[5])

    def p_join_argument_list(self, p):
        'join_argument : expression COMMA LPAREN column_ref_list RPAREN'
        p[0] = JoinTarget(p[1], p[4])

    def p_join_argument_single(self, p):
        'join_argument : expression COMMA column_ref'
        p[0] = JoinTarget(p[1], [p[3]])

    # column_ref refers to the name or position of a column; these serve
    # as arguments to join.
    def p_column_ref_list(self, p):
        '''column_ref_list : column_ref_list COMMA column_ref
                           | column_ref'''
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    def p_column_ref_string(self, p):
        'column_ref : ID'
        p[0] = p[1]

    def p_column_ref_index(self, p):
        'column_ref : DOLLAR INTEGER_LITERAL'
        p[0] = p[2]

    def p_expression_filter(self, p):
        'expression : FILTER ID BY sexpr'
        p[0] = ('FILTER', p[2], p[4])

    # scalar expressions map to raco.Expression instances; these are operations
    # that return scalar types.

    def p_sexpr_integer_literal(self, p):
        'sexpr : INTEGER_LITERAL'
        p[0] = sexpr.NumericLiteral(p[1])

    def p_sexpr_string_literal(self, p):
        'sexpr : STRING_LITERAL'
        p[0] = sexpr.StringLiteral(p[1])

    def p_sexpr_id(self, p):
        'sexpr : ID'
        p[0] = sexpr.NamedAttributeRef(p[1])

    def p_sexpr_index(self, p):
        'sexpr : DOLLAR INTEGER_LITERAL'
        p[0] = sexpr.UnnamedAttributeRef(p[2])

    def p_sexpr_id_dot_id(self, p):
        'sexpr : ID DOT ID'
        p[0] = sexpr.DottedAttributeRef(p[1], p[3])

    def p_sexpr_id_dot_pos(self, p):
        'sexpr : ID DOT DOLLAR INTEGER_LITERAL'
        p[0] = sexpr.DottedAttributeRef(p[1], p[4])

    def p_sexpr_group(self, p):
        'sexpr : LPAREN sexpr RPAREN'
        p[0] = p[2]

    def p_sexpr_uminus(self, p):
        'sexpr : MINUS sexpr %prec UMINUS'
        p[0] = sexpr.TIMES(sexpr.NumericLiteral(-1), p[2])

    def p_sexpr_binop(self, p):
        '''sexpr : sexpr PLUS sexpr
                   | sexpr MINUS sexpr
                   | sexpr TIMES sexpr
                   | sexpr DIVIDE sexpr
                   | sexpr GT sexpr
                   | sexpr LT sexpr
                   | sexpr GE sexpr
                   | sexpr LE sexpr
                   | sexpr NE sexpr
                   | sexpr EQ sexpr
                   | sexpr AND sexpr
                   | sexpr OR sexpr'''
        p[0] = binops[p[2]](p[1], p[3])

    def p_sexpr_not(self, p):
        'sexpr : NOT sexpr'
        p[0] = sexpr.NOT(p[2])

    def p_sexpr_unary_aggregate(self, p):
        'sexpr : unary_aggregate_func LPAREN sexpr RPAREN'
        p[0] = p[1](p[3])

    def p_unary_aggregate_func(self, p):
        '''unary_aggregate_func : MAX
                                | MIN
                                | SUM
                                | COUNT'''

        # TODO: support average once we have floating point
        if p[1] == 'MAX': func = sexpr.MAX
        if p[1] == 'MIN': func = sexpr.MIN
        if p[1] == 'SUM': func = sexpr.SUM
        if p[1] == 'COUNT': func = sexpr.COUNT

        p[0] = func

    def p_sexpr_unbox_implicit(self, p):
        'sexpr : TIMES ID'
        p[0] = sexpr.Unbox(p[2], None)

    def p_sexpr_unbox_explicit_name(self, p):
        'sexpr : TIMES ID DOT ID'
        p[0] = sexpr.Unbox(p[2], p[4])

    def p_sexpr_unbox_explicit_pos(self, p):
        'sexpr : TIMES ID DOT DOLLAR INTEGER_LITERAL'
        p[0] = sexpr.Unbox(p[2], p[5])

    def p_empty(self, p):
        'empty :'
        pass

    def parse(self, s):
        parser = yacc.yacc(module=self, debug=True)
        return parser.parse(s, lexer=scanner.lexer, tracking=True)

    def p_error(self, p):
        self.log.error("Syntax error: %s", str(p))
