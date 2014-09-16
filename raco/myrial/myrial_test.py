import json
import unittest

from raco.language.myrialang import compile_to_json
import raco.fakedb
from raco.language.myrialang import MyriaLeftDeepTreeAlgebra
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
import raco.viz
from raco.replace_with_repr import replace_with_repr


class MyrialTestCase(unittest.TestCase):
    def create_db(self):
        return raco.fakedb.FakeDatabase()

    def setUp(self):
        self.db = self.create_db()

        self.parser = parser.Parser()
        self.processor = interpreter.StatementProcessor(self.db)

    def parse(self, query):
        """Parse a query"""
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)

    def get_plan(self, query, **kwargs):
        """Get the MyriaL query plan for a query"""
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        if kwargs.get('logical', False):
            p = self.processor.get_logical_plan(**kwargs)
        else:
            p = self.processor.get_physical_plan(**kwargs)
        # verify that we can stringify p
        # TODO verify the string somehow?
        assert str(p)
        # verify that we can convert p to a dot
        # TODO verify the dot somehow?
        raco.viz.get_dot(p)
        # Test repr
        return replace_with_repr(p)

    def get_logical_plan(self, query, **kwargs):
        """Get the logical plan for a MyriaL query"""
        kwargs['logical'] = True
        return self.get_plan(query, **kwargs)

    def get_physical_plan(self, query, **kwargs):
        """Get the physical plan for a MyriaL query"""
        kwargs['logical'] = False
        return self.get_plan(query, **kwargs)

    def execute_query(self, query, test_logical=False, skip_json=False,
                      output='OUTPUT'):
        """Run a test query against the fake database"""
        plan = self.get_plan(query, logical=test_logical)

        if not test_logical and not skip_json:
            # Test that JSON compilation runs without error
            # TODO: verify the JSON output somehow?
            json_string = json.dumps(compile_to_json(
                "some query", "some logical plan", plan, "myrial"))
            assert json_string

        self.db.evaluate(plan)

        return self.db.get_table(output)

    def check_result(self, query, expected, test_logical=False,
                     skip_json=False, output='OUTPUT', scheme=None):
        """Execute a test query with an expected output"""
        actual = self.execute_query(query, test_logical, skip_json, output)
        self.assertEquals(actual, expected)

        if scheme:
            self.assertEquals(self.db.get_scheme(output), scheme)

    def check_scheme(self, query, scheme):
        """Execute a test query with an expected output schema."""
        actual = self.execute_query(query)
        self.assertEquals(self.db.get_scheme('OUTPUT'), scheme)
