import unittest
import json

import raco.fakedb
from raco import RACompiler
from raco.language import MyriaAlgebra
from myrialang import compile_to_json


class DatalogTestCase(unittest.TestCase):

    def setUp(self):
        self.db = raco.fakedb.FakeDatabase()

    def execute_query(self, query):
        '''Run a test query against the fake database'''

        #print query

        dlog = RACompiler()
        dlog.fromDatalog(query)

        #print dlog.logicalplan

        dlog.optimize(target=MyriaAlgebra,
                      eliminate_common_subexpressions=False)

        #print dlog.physicalplan

        # test whether we can generate json without errors
        json_string = json.dumps(compile_to_json(
            "", dlog.logicalplan, dlog.physicalplan))
        assert json_string

        op = dlog.physicalplan[0][1]
        output_op = raco.algebra.StoreTemp('__OUTPUT__', op)
        self.db.evaluate(output_op)
        return self.db.get_temp_table('__OUTPUT__')

    def run_test(self, query, expected):
        '''Execute a test query with an expected output'''
        actual = self.execute_query(query)
        self.assertEquals(actual, expected)
