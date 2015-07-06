import unittest
import json

import raco.fakedb
from raco import RACompiler
from raco.backends.myria import (compile_to_json,
                                 MyriaLeftDeepTreeAlgebra,
                                 MyriaHyperCubeAlgebra)
from raco.catalog import FakeCatalog


class DatalogTestCase(unittest.TestCase):

    def setUp(self):
        self.db = raco.fakedb.FakeDatabase()

    def execute_query(self, query, test_logical=False, skip_json=False,
                      output="OUTPUT", algebra=MyriaLeftDeepTreeAlgebra):
        """Run a test query against the fake database"""

        dlog = RACompiler()
        dlog.fromDatalog(query)

        assert algebra in [MyriaLeftDeepTreeAlgebra,
                           MyriaHyperCubeAlgebra]

        if test_logical:
            plan = dlog.logicalplan
        else:
            if algebra == MyriaLeftDeepTreeAlgebra:
                dlog.optimize(MyriaLeftDeepTreeAlgebra())
            else:
                dlog.optimize(MyriaHyperCubeAlgebra(FakeCatalog(64)))
            plan = dlog.physicalplan

            if not skip_json:
                # test whether we can generate json without errors
                json_string = json.dumps(compile_to_json(
                    query, dlog.logicalplan, dlog.physicalplan, "datalog"))
                assert json_string

        self.db.evaluate(plan)
        return self.db.get_table(output)

    def check_result(self, query, expected, test_logical=False,
                     skip_json=False, output="OUTPUT",
                     algebra=MyriaLeftDeepTreeAlgebra):
        """Execute a test query with an expected output"""
        actual = self.execute_query(query, test_logical=test_logical,
                                    skip_json=skip_json, output=output,
                                    algebra=algebra)
        self.assertEquals(actual, expected)
