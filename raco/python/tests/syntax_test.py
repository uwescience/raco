# coding=utf-8
""" Tests for expressions with invalid syntax """

from raco.python import convert
from raco.python.exceptions import PythonConvertException, \
    PythonSyntaxException, PythonArgumentException, \
    PythonUnsupportedOperationException
from python_test import PythonTestCase


class TestSyntax(PythonTestCase):
    def test_invalid_attribute(self):
        query = """lambda t: t.foo == 6"""
        self.assertRaises(PythonConvertException,
                          lambda: convert(query, [self.schema]))

    def test_invalid_dotted_attribute(self):
        query = """lambda t: t[0].foo == 6"""
        self.assertRaises(PythonUnsupportedOperationException,
                          lambda: convert(query, [self.schema]))

    def test_syntax_error(self):
        query = """lambda t: t[0] = 6"""
        self.assertRaises(PythonSyntaxException,
                          lambda: convert(query, [self.schema]))

    def test_undefined_token(self):
        query = """lambda t: foo[0] == 6"""
        self.assertRaises(PythonConvertException,
                          lambda: convert(query, [self.schema]))

    def test_out_of_range_index(self):
        query = """lambda t: t[999] == 6"""
        self.assertRaises(PythonConvertException,
                          lambda: convert(query, [self.schema]))

    def test_negative_index(self):
        query = """lambda t: t[-1] == 6"""
        self.assertRaises(PythonConvertException,
                          lambda: convert(query, [self.schema]))

    def test_slice(self):
        query = """lambda t: t[1:2] == 6"""
        self.assertRaises(PythonConvertException,
                          lambda: convert(query, [self.schema]))

    def test_substring_negative_index(self):
        query = """lambda t: t.id[-5] == 'x'"""
        self.assertRaises(PythonConvertException,
                          lambda: convert(query, [self.schema]))

    def test_substring_step(self):
        query = """lambda t: t.id[1:10:3] == 'x'"""
        self.assertRaises(PythonConvertException,
                          lambda: convert(query, [self.schema]))

    def test_unrecognized_function(self):
        query = """lambda t: foo(t.id) == 1"""
        self.assertRaises(PythonConvertException,
                          lambda: convert(query, [self.schema]))

    def test_unrecognized_keyword(self):
        query = """lambda t: t.id == foo"""
        self.assertRaises(PythonConvertException,
                          lambda: convert(query, [self.schema]))

    def test_mismatched_parenthesis(self):
        query = """lambda t: (t[999] == 6"""
        self.assertRaises(PythonConvertException,
                          lambda: convert(query, [self.schema]))

    def test_partial_keyword(self):
        query = """foolambda t: t[999] == 6"""
        self.assertRaises(PythonConvertException,
                          lambda: convert(query, [self.schema]))
