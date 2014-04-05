import unittest
from testquery import testquery
from example_queries import testEmit
from raco.language import CCAlgebra


class ClangTest(unittest.TestCase):
    def check(self, query, name):
        testEmit(query, name, CCAlgebra)
        self.assertTrue(testquery(name))

    def test_scan(self):
        self.check("A(s1) :- T1(s1)", "scan")

    def test_select(self):
        self.check("A(s1) :- T1(s1), s1>10", "select") 

    def test_join(self):
        self.check("A(s1,o2) :- T3(s1,p1,o1), R3(o2,p1,o2)", "join")
            
    def test_select_conjunction(self):
        self.check("A(s1) :- T1(s1), s1>0, s1<10", "select_conjunction"),


if __name__ == '__main__':
    unittest.main()
