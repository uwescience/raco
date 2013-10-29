
import collections
import math
import unittest

import raco.fakedb
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
from raco.myrialang import compile_to_json

class MyrialTestCase(unittest.TestCase):

    def setUp(self):
        self.db = raco.fakedb.FakeDatabase()
        self.parser = parser.Parser()
        self.processor = interpreter.StatementProcessor(self.db)

    def execute_query(self, query, test_logical=False):
        '''Run a test query against the fake database'''
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)

        if test_logical:
            plan = self.processor.get_logical_plan()
        else:
            plan = self.processor.get_physical_plan()
            json = compile_to_json(query, '', [('A', plan)])

        self.db.evaluate(plan)


        return self.db.get_temp_table('__OUTPUT0__')

    def run_test(self, query, expected, test_logical=False):
        '''Execute a test query with an expected output'''
        actual = self.execute_query(query, test_logical)
        self.assertEquals(actual, expected)

