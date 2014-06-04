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

    def parse(self, query):
        '''Parse a query'''
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)

    def get_plan(self, query, logical=False, multiway_join=False):
        '''Get the MyriaL query plan for a query'''
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        if logical:
            return self.processor.get_logical_plan()
        else:
            if multiway_join:
                self.processor.multiway_join = True
            else:
                self.processor.multiway_join = False
            return self.processor.get_physical_plan()

    def get_logical_plan(self, query):
        '''Get the logical plan for a MyriaL query'''
        return self.get_plan(query, True)

    def get_physical_plan(self, query):
        '''Get the physical plan for a MyriaL query'''
        return self.get_plan(query, False)

    def execute_query(self, query, test_logical=False, skip_json=False,
                      output='OUTPUT'):
        '''Run a test query against the fake database'''
        plan = self.get_plan(query, test_logical)

        if not test_logical and not skip_json:
            # Test that JSON compilation runs without error
            # TODO: verify the JSON output somehow?
            json_string = json.dumps(compile_to_json(
                "some query", "some logical plan", plan))
            assert json_string

        self.db.evaluate(plan)

        return self.db.get_table(output)

    def check_result(self, query, expected, test_logical=False,
                     skip_json=False, output='OUTPUT'):
        '''Execute a test query with an expected output'''
        actual = self.execute_query(query, test_logical, skip_json, output)
        self.assertEquals(actual, expected)
