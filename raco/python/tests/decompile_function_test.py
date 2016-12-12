# coding=utf-8

""" Tests for decompiling lambdas """

import unittest

from raco.python.util import decompile


class TestDecompileFunctions(unittest.TestCase):
    def test_simple(self):
        def f():
            def g(): return 0
            return g
        s = decompile.get_source(f())
        exec s
        self.assertEquals(g(), 0)

    def test_space_before_colon(self):
        def f() :
            def g(): return 0
            return g
        s = decompile.get_source(f())
        exec s
        self.assertEquals(g(), 0)

    def test_multiple_lines(self):
        def f() :
            def g():

                return 0
            return g
        s = decompile.get_source(f())
        exec s
        self.assertEquals(g(), 0)

    def test_line_continuation(self):
        def f() :
            def g():
                return \
                    0
            return g
        s = decompile.get_source(f())
        exec s
        self.assertEquals(g(), 0)

    def test_commants(self):
        def f() :
            def g(): # foo
                return \
                    0 # bar
            return g
        s = decompile.get_source(f())
        exec s
        self.assertEquals(g(), 0)

    def test_variable(self):
        def f():
            def g(): return 0
            return g
        h = f()
        s = decompile.get_source(h)
        exec s
        self.assertEquals(g(), 0)

    def test_function_with_lambda(self):
        def f():
            def g(x=lambda: 0): return x()
            return g
        s = decompile.get_source(f())
        exec s
        self.assertEquals(g(), 0)

    def test_embedded_lambda_token(self):
        def f():
            def g(x="lambda: 0"): return "lambda: 1"
            return g
        s = decompile.get_source(f())
        exec s
        self.assertEquals(g(), "lambda: 1")

    def test_parameters(self):
        def f():
            def g(x, y, z=1): return x + y + z
            return g
        s = decompile.get_source(f())
        exec s
        self.assertEquals(g(3, 4, 5), 12)
        self.assertEquals(g(3, 4), 8)

    def test_args_kwargs(self):
        def f():
            def g(*args, **kwargs): return args[0] + kwargs['foo']
            return g
        s = decompile.get_source(f())
        exec s
        self.assertEquals(g(3, foo=4), 7)

    def _member(self): return 1

    def test_member(self):
        s = decompile.get_source(self._member)
        exec s
        self.assertEquals(_member(self), 1)

    @classmethod
    def _classmethod(self): return 1

    def test_classmethod(self):
        s = decompile.get_source(self._classmethod)
        exec s
        self.assertEquals(_classmethod(self), 1)

    @staticmethod
    def _staticmethod(x): return x + 1

    def test_staticmethod(self):
        s = decompile.get_source(self._staticmethod)
        exec s
        self.assertEquals(_staticmethod(3), 4)

    def test_unpacking(self):
        def fff((x, y)): return x + y
        s = decompile.get_source(fff)
        exec s.replace('fff', 'ggg')
        self.assertEquals(ggg((1, 2)), 3)

