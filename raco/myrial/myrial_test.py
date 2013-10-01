
import collections
import math
import unittest

import raco.fakedb
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser

class MyrialTestCase(unittest.TestCase):

    def setUp(self):
        self.db = raco.fakedb.FakeDatabase()
        self.parser = parser.Parser()
        self.processor = interpreter.StatementProcessor(self.db)

    def execute_query(self, query):
        '''Run a test query against the fake database'''
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)

        _var, op = self.processor.output_symbols[0]
        return self.db.evaluate_to_bag(op)

    def run_test(self, query, expected):
        '''Execute a test query with an expected output'''
        actual = self.execute_query(query)
        self.assertEquals(actual, expected)

