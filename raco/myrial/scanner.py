#!/usr/bin/python

import ply.lex as lex

# identifiers with special meaning; case-insensitive:
# LoAd, load, LOAD are all accepted
reserved = ['LOAD', 'STORE', 'LIMIT', 'SHUFFLE', 'SEQUENCE', 'CROSS', 'JOIN',
            'GROUP','EMIT', 'AS', 'DIFF', 'UNIONALL', 'INTERSECT', 'APPLY',
            'DUMP', 'FILTER', 'TABLE', 'ORDER', 'ASC', 'DESC', 'BY', 'WHILE',
            'INT', 'STRING', 'DESCRIBE', 'DO', 'EXPLAIN', 'DISTINCT', 'SCAN',
            'COUNT']

# Token types; required by ply to have this variable name
tokens = ['LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET', 'PLUS', 'MINUS', 'TIMES',
          'DIVIDE', 'MOD', 'LOR', 'LAND', 'LNOT', 'LT', 'GT', 'GE', 'LE', 'EQ',
          'NE', 'COMMA', 'SEMI', 'EQUALS', 'COLON', 'DOLLAR', 'DOT', 'ID',
          'STRING_LITERAL', 'INTEGER_LITERAL', 'LBRACE', 'RBRACE'] + reserved

# Regular expression rules for simple tokens
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_LBRACE           = r'\{'
t_RBRACE           = r'\}'
t_PLUS  = r'\+'
t_MINUS  = r'-'
t_TIMES  = r'\*'
t_DIVIDE = r'/'
t_MOD   = r'%'

t_LOR              = r'\|\|'
t_LAND             = r'&&'
t_LNOT             = r'!'
t_LT               = r'<'
t_GT               = r'>'
t_LE               = r'<='
t_GE               = r'>='
t_EQ               = r'=='
t_NE               = r'!='

t_COMMA = r','
t_SEMI = r';'
t_EQUALS = r'='
t_COLON = r':'
t_DOLLAR = r'\$'
t_DOT = r'\.'

# Regular expressions for non-trivial tokens

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    global reserved

    upped = t.value.upper()
    if upped in reserved:
        t.type = upped
        return t
    else:
        t.type = 'ID'
        return t

def t_INTEGER_LITERAL(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_STRING_LITERAL(t):
    r'"([^\\\n"]|\\.)*"'
    t.value=t.value[1:-1].decode("string_escape")
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# C-style comments
def t_c_comment(t):
    r'/\*(.|\n)*?\*/'
    t.lexer.lineno += t.value.count('\n')

# database-style comments
def t_db_comment(t):
    r'--.*'

# Always ignore whitespace (spaces and tabs)
t_ignore  = ' \t\v'

# Error handling rule
def t_error(t):
    print "Illegal character token: " + str(t)

lexer = lex.lex()
