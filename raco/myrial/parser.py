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

# ID is a symbol name that identifies an input expression; columns is a list of
# columns expressed as either names or integer positions.
JoinTarget = collections.namedtuple('JoinTarget',['expr', 'columns'])

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
    'AND' : colexpr.AND,
    'OR' : colexpr.OR,
}

class Parser:
    def __init__(self, log=yacc.PlyLogger(sys.stderr)):
        self.log = log
        self.tokens = scanner.tokens

        # Precedence among column expression operators in ascending order; this
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
        'statement : STORE LPAREN ID COMMA relation_key RPAREN SEMI'
        p[0] = ('STORE', p[3], p[5])

    def p_expression_id(self, p):
        'expression : ID'
        p[0] = ('ALIAS', p[1])

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

    def p_expression_bag_comp(self, p):
        'expression : LBRACKET FROM expression opt_where_clause \
        emit_clause RBRACKET'
        p[0] = ('BAGCOMP', p[3], p[4], p[5])

    def p_opt_where_clause(self, p):
        '''opt_where_clause : WHERE colexpr
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
        '''emit_arg : string_arg EQUALS colexpr
                    | colexpr'''
        if len(p) == 4:
            p[0] = (p[1], p[3])
        else:
            p[0] = (None, p[1])

    def p_expression_limit(self, p):
        'expression : LIMIT ID COMMA INTEGER_LITERAL'
        p[0] = ('LIMIT', p[2], p[4])

    def p_expression_distinct(self, p):
        'expression : DISTINCT LPAREN expression RPAREN'
        p[0] = ('DISTINCT', p[3])

    def p_expression_countall(self, p):
        'expression : COUNTALL ID'
        p[0] = ('COUNTALL', p[2])

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
        'expression : FILTER ID BY colexpr'
        p[0] = ('FILTER', p[2], p[4])

    # column expressions map to raco.Expression instances; these are operations
    # that return atomic types, and that are suitable as arguments for apply
    # and fiter

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
        p[0] = colexpr.TIMES(colexpr.NumericLiteral(-1), p[2])

    def p_expression_binop(self, p):
        '''colexpr : colexpr PLUS colexpr
                   | colexpr MINUS colexpr
                   | colexpr TIMES colexpr
                   | colexpr DIVIDE colexpr
                   | colexpr GT colexpr
                   | colexpr LT colexpr
                   | colexpr GE colexpr
                   | colexpr LE colexpr
                   | colexpr NE colexpr
                   | colexpr EQ colexpr
                   | colexpr AND colexpr
                   | colexpr OR colexpr'''
        p[0] = binops[p[2]](p[1], p[3])

    def p_colexpr_not(self, p):
        'colexpr : NOT colexpr'
        p[0] = colexpr.NOT(p[2])

    def p_empty(self, p):
        'empty :'
        pass

    def parse(self, s):
        parser = yacc.yacc(module=self, debug=True)
        return parser.parse(s, lexer=scanner.lexer, tracking=True)

    def p_error(self, p):
        self.log.error("Syntax error: %s", str(p))
