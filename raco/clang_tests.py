import unittest
import os
import sys
sys.path.append('./examples')
from emitcode import emitCode
from raco.language import CCAlgebra, GrappaAlgebra


class ClangEmitTest(unittest.TestCase):
    def check(self, query, name):
        emitCode(query, name, CCAlgebra())

        fn = name + ".cpp"

        # only checks that file exists
        self.assertTrue(os.path.isfile(fn))
        self.assertGreater(os.stat(fn).st_size, 0)
        os.remove(fn)

    def test_self_join(self):
        self.check("A(a,b) :- R2(a,b), R2(a,c)", "self_join")


class GrappaEmitTest(unittest.TestCase):
    def check(self, query, name):
        emitCode(query, name, CCAlgebra())

        fn = name + ".cpp"

        # only checks that file exists
        self.assertTrue(os.path.isfile(fn))
        self.assertGreater(os.stat(fn).st_size, 0)
        os.remove(fn)

    def test_self_join(self):
        self.check("A(a,b) :- R2(a,b), R2(a,c)", "self_join")
