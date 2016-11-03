# coding=utf-8

""" Tests for the 'convert' entry-point function """
import ast
import unittest

from raco.expression import NumericLiteral
from raco.python import convert


class TestConvert(unittest.TestCase):
    def test_string(self):
        f = "lambda: 0"
        e = convert(f, None)
        self.assertEqual(e, NumericLiteral(0))

    def test_ast(self):
        t = ast.parse("lambda: 0")
        e = convert(t, None)
        self.assertEqual(e, NumericLiteral(0))

    def test_lambda(self):
        f = lambda: 0
        e = convert(f, None)
        self.assertEqual(e, NumericLiteral(0))

    def test_function(self):
        def f():
            return 0
        e = convert(f, None)
        self.assertEqual(e, NumericLiteral(0))
