import json
import unittest

from raco.myrialang import compile_to_json
import raco.fakedb
from raco.myrial.myrial_parser import *


class MyrialTestCase(unittest.TestCase):

    def setUp(self):
        self.db = raco.fakedb.FakeDatabase()

    def parse(self, query):
        '''Parse a query'''
        return myrial_to_ast(query, catalog=self.db)

    def get_plan(self, query, logical=False):
        '''Get the MyriaL query plan for a query'''
        return myrial_to_plan(query, catalog=self.db, logical=logical)

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
                "some query", "some logical plan", plan, self.db))
            assert json_string

        self.db.evaluate(plan)

        return self.db.get_table(output)

    def check_result(self, query, expected, test_logical=False,
                     skip_json=False, output='OUTPUT', scheme=None):
        '''Execute a test query with an expected output'''
        actual = self.execute_query(query, test_logical, skip_json, output)
        self.assertEquals(actual, expected)
        if scheme:
            self.assertEquals(self.db.get_scheme(output), scheme)

    def check_scheme(self, query, scheme):
        '''Execute a test query with an expected output schema.'''
        actual = self.execute_query(query)
        self.assertEquals(self.db.get_scheme('OUTPUT'), scheme)
