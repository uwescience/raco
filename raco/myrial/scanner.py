#!/usr/bin/python
# -*- coding: UTF-8 -*-

import ply.lex as lex

import raco.myrial.exceptions

keywords = ['WHILE', 'DO', 'DEF', 'APPLY', 'CASE', 'WHEN', 'THEN',
            'ELSE', 'END', 'CONST', 'LOAD', 'DUMP', 'CSV', 'SCHEMA',
            'OPP', 'TIPSY', 'UDA', 'TRUE', 'FALSE', 'HASH', 'BROADCAST',
            'ROUND_ROBIN', 'UNTIL', 'CONVERGENCE', "SYNC", "ASYNC",
            'ALTERNATE', 'PULL_IDB', 'PULL_EDB', 'BUILD_EDB', 'ASC', 'DESC']

types = ['INT', 'STRING', 'FLOAT', 'BOOLEAN', 'BLOB']

comprehension_keywords = ['SELECT', 'AS', 'EMIT', 'FROM', 'WHERE', 'ORDERBY']

word_operators = ['AND', 'OR', 'NOT']

builtins = ['EMPTY', 'WORKER_ID', 'SCAN', 'COUNTALL', 'COUNT', 'STORE',
            'EXPORT', 'DIFF', 'CROSS', 'JOIN', 'UNION', 'UNIONALL',
            'INTERSECT', 'DISTINCT', 'LIMIT', 'SINK', 'SAMPLESCAN', 'LIKE']


# identifiers with special meaning; case-insensitive
reserved = (keywords + types + comprehension_keywords
            + word_operators + builtins)

# Token types; required by ply to have this variable name

tokens = ['LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET', 'DOT', 'PLUS', 'MINUS',
          'TIMES', 'DIVIDE', 'IDIVIDE', 'MOD', 'LT', 'GT', 'GE', 'GE2',
          'LE', 'LE2', 'EQ', 'NE', 'NE2', 'NE3', 'COMMA', 'SEMI', 'EQUALS',
          'COLON', 'DOLLAR', 'ID', 'LARROW',
          'STRING_LITERAL', 'INTEGER_LITERAL', 'FLOAT_LITERAL', 'BLOB_LITERAL',
          'LBRACE', 'RBRACE'] + reserved

# Regular expression rules for simple tokens
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_LBRACE = r'\{'
t_RBRACE = r'\}'

t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_IDIVIDE = r'//'
t_MOD = r'%'

t_LT = r'<'
t_GT = r'>'
t_LE = r'<='
t_LE2 = u'≤'
t_GE = r'>='
t_GE2 = u'≥'
t_EQ = r'=='
t_NE = r'!='
t_NE2 = r'<>'
t_NE3 = u'≠'

t_DOT = r'\.'
t_COMMA = r','
t_SEMI = r';'
t_EQUALS = r'='
t_COLON = r':'
t_DOLLAR = r'\$'
t_LARROW = r'<-'
# Regular expressions for non-trivial tokens


def t_ORDERBY(t):
    r'(?i)ORDER\s+BY'
    return t


def t_BLOB_LITERAL(t):
    r'b((\'(\\x[0-9a-fA-F]{2})*\')|(\"(\\x[0-9a-fA-F]{2})*\"))'
    t.value = bytes(t.value[2:-1].decode("string_escape"))
    return t


def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    global reserved
    upped = t.value.upper()
    if upped in reserved:
        t.type = upped
        t.value = upped
        return t
    else:
        t.type = 'ID'
        return t


def t_FLOAT_LITERAL(t):
    # Left group: scientific notation, right group: decimal form
    r"""(\d*(\.\d+)?[eE][-+]?\d+)|(\d*\.\d+)"""
    t.value = float(t.value)
    return t


def t_INTEGER_LITERAL(t):
    r'\d+'
    t.value = int(t.value)
    return t


def t_STRING_LITERAL(t):
    r'"([^\\\n"]|\\.)*"|\'([^\\\n\']|\\.)*\''
    t.value = t.value[1:-1].decode("string_escape")
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
t_ignore = ' \t\v'


# Error handling rule
def t_error(t):
    raise raco.myrial.exceptions.MyrialScanException(t)


lexer = lex.lex()
