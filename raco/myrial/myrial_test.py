import json
import unittest

from raco.myrialang import compile_to_json
import raco.fakedb
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser


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

            # Test that JSON compilation runs without error
            # TODO: verify the JSON output somehow?
            json_string = json.dumps(compile_to_json(
                "some query", "some logical plan", plan))
            assert json_string

        self.db.evaluate(plan)

        return self.db.get_table('OUTPUT')

    def check_result(self, query, expected, test_logical=False):
        '''Execute a test query with an expected output'''
        actual = self.execute_query(query, test_logical)
        self.assertEquals(actual, expected)
