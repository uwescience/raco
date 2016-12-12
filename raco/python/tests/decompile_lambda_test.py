# coding=utf-8

""" Tests for decompiling lambdas """

import unittest

from raco.python.exceptions import PythonConvertException
from raco.python.util import decompile


class TestDecompileLambdas(unittest.TestCase):
    def test_simple(self):
        s = decompile.get_source(lambda: 0)
        self.assertEquals(eval(s)(), 0)

    def test_no_space(self):
        s = decompile.get_source(lambda:0)
        self.assertEquals(eval(s)(), 0)

    def test_space_after_colon(self):
        s = decompile.get_source(lambda :0)
        self.assertEquals(eval(s)(), 0)

    def test_variable(self):
        f = lambda: 0
        s = decompile.get_source(f)
        self.assertEquals(eval(s)(), f())

    def test_newline(self):
        f = \
            lambda: 0
        s = decompile.get_source(f)
        self.assertEquals(eval(s)(), f())

    def test_newline2(self):
        f = lambda: \
            0
        s = decompile.get_source(f)
        self.assertEquals(eval(s)(), f())

    def test_tuple(self):
        t = (lambda: 0), 5
        s = decompile.get_source(t[0])
        self.assertEquals(eval(s)(), t[0]())

    def test_tuple2(self):
        f = lambda :  (0, 5)
        s = decompile.get_source(f)
        self.assertEquals(eval(s)(), f())

    def test_multiple_lambdas(self):
        t = lambda: 1, lambda: 2, lambda: 3
        for f in t:
            s = decompile.get_source(f)
            self.assertEquals(eval(s)(), f())

    def test_embedded_lambda_token(self):
        f = lambda: "lambda: 0"
        s = decompile.get_source(f)
        self.assertEquals(eval(s)(), f())

    def test_parameters(self):
        f = lambda x: x
        s = decompile.get_source(f)
        self.assertEquals(eval(s)(5), f(5))

    def test_multiple_parameters(self):
        f = lambda x, y: x + y
        s = decompile.get_source(f)
        self.assertEquals(eval(s)(5, 6), f(5, 6))

    def test_args_kwargs(self):
        f = lambda *args, **kwargs: args[0] + kwargs['foo']
        s = decompile.get_source(f)
        self.assertEqual(eval(s)(5, foo=6), f(5, foo=6))

    def test_unpacking(self):
        """ Unpacking is not currently supported """
        f = lambda (x, y): x + y
        self.assertRaises(PythonConvertException,
                          lambda: decompile.get_source(f))
