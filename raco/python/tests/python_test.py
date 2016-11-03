# coding=utf-8
""" Base class for Python lambda-oriented unit tests"""

import json
import unittest

import raco.fakedb
import raco.viz
import raco.myrial.interpreter as interpreter
from raco.backends.myria import compile_to_json
from raco.fake_data import FakeData
from raco import relation_key
from raco.replace_with_repr import replace_with_repr
from raco.algebra import Sequence, Scan, Store
from raco import compile
from raco.backends.logical import OptLogicalAlgebra
from raco.backends.myria import MyriaLeftDeepTreeAlgebra


class PythonTestCase(unittest.TestCase, FakeData):
    def setUp(self):
        self.db = raco.fakedb.FakeDatabase()
        self.processor = interpreter.StatementProcessor(self.db)
        self.db.ingest(self.emp_key, self.emp_table, self.emp_schema)
        self.input = relation_key.RelationKey.from_string(self.emp_key)
        self.schema = self.emp_schema
        self.scan = Scan(self.input, self.schema)
        self.output = relation_key.RelationKey.from_string('OUTPUT')

    def get_query(self, expression):
        return Sequence([Store(self.output, expression)])

    def get_plan(self, statements, logical=False):
        """Get the query plan"""
        self.logical = compile.optimize(statements, OptLogicalAlgebra())
        self.physical = compile.optimize(self.logical,
                                         MyriaLeftDeepTreeAlgebra())
        plan = self.logical if logical else self.physical

        # Verify that we can stringify
        assert str(plan)
        # Verify that we can convert to a dot
        raco.viz.get_dot(plan)
        # verify repr
        return replace_with_repr(plan)

    def execute_query(self, query, test_logical=False, skip_json=False):
        """Run a test query against the fake database"""
        plan = self.get_plan(query, logical=test_logical)

        if not test_logical and not skip_json:
            # Test that JSON compilation runs without error
            json_string = json.dumps(compile_to_json("some query",
                                                     "some logical plan",
                                                     plan,
                                                     "myrial"))
            assert json_string

        self.db.evaluate(plan)

        return self.db.get_table(self.output)

    def check_result(self, query, expected, test_logical=False,
                     skip_json=False, scheme=None):
        """Execute a test query with an expected output"""
        actual = self.execute_query(query, test_logical, skip_json)
        self.assertEquals(actual, expected)

        if scheme:
            self.assertEquals(self.db.get_scheme(self.output), scheme)

    def check_scheme(self, query, scheme):
        """Execute a test query with an expected output schema."""
        self.execute_query(query)
        self.assertEquals(self.db.get_scheme(self.output), scheme)
