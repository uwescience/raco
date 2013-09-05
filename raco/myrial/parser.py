#!/usr/bin/env python

import ply.yacc as yacc

import raco.myrial.scanner as scanner
import raco.scheme as scheme

import collections
import sys

class JoinColumnCountMismatchException(Exception):
    pass

class ParseException(Exception):
    pass

# ID is a symbol name; columns is list containing either column names
# (as strings) or integer offsets (starting at zero).
JoinTarget = collections.namedtuple('JoinTarget',['id', 'columns'])

class RelationKey(object):
    def __init__(self, table, program='default', user='nobody'):
        self.table = table
        self.program = program
        self.user = user

    def __repr__(self):
        return 'RelationKey(%s,%s,%s)' % (self.table, self.program,self.user)

class Parser:
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
                 | UNION'''
        p[0] = p[1]

    def p_expression_join(self, p):
        'expression : JOIN join_argument COMMA join_argument'
        if len(p[2].columns) != len(p[4].columns):
            raise JoinColumnCountMismatchException()
        p[0] = ('JOIN', p[2], p[4])

    def p_join_argument_list(self, p):
        'join_argument : ID BY LPAREN column_arg_list RPAREN'
        p[0] = JoinTarget(p[1], p[4])

    def p_join_argument_single(self, p):
        'join_argument : ID BY column_arg'
        p[0] = JoinTarget(p[1], [p[3]])

    def p_column_arg_list(self, p):
        '''column_arg_list : column_arg_list COMMA column_arg
                           | column_arg'''
        if len(p) == 4:
            cols = p[1] + [p[3]]
        else:
            cols = [p[1]]
        p[0] = cols

    def p_column_arg_id(self, p):
        'column_arg : ID'
        p[0] = p[1]

    def p_column_arg_index(self, p):
        'column_arg : DOLLAR INTEGER_LITERAL'
        p[0] = p[2]

    def p_empty(self, p):
        'empty :'
        pass

    def parse(self, s):
        parser = yacc.yacc(module=self, debug=True)
        return parser.parse(s, lexer=scanner.lexer, tracking=True)

    def p_error(self, p):
        self.log.error("Syntax error: %s" %  str(p))
