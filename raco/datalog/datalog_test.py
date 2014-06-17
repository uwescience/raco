import unittest
import json

import raco.fakedb
from raco import RACompiler
from raco.language import MyriaAlgebra
from raco.myrialang import compile_to_json
from raco.relation_key import RelationKey


class DatalogTestCase(unittest.TestCase):

    def setUp(self):
        self.db = raco.fakedb.FakeDatabase()

    def execute_query(self, query, name):
        """Run a test query against the fake database"""

        dlog = RACompiler()
        dlog.fromDatalog(query)

        dlog.optimize(target=MyriaAlgebra,
                      eliminate_common_subexpressions=False)

        # test whether we can generate json without errors
        json_string = json.dumps(compile_to_json(
            query, dlog.logicalplan, dlog.physicalplan))
        assert json_string

        self.db.evaluate(dlog.physicalplan)
        return self.db.get_table(name)

    def check_result(self, query, expected, name):
        """Execute a test query with an expected output"""
        actual = self.execute_query(query, name)
        self.assertEquals(actual, expected)
