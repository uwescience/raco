import unittest
import csv

from raco.compile import compile
from raco.clangtestdb import ClangTestDatabase
from raco.language.clang import CCAlgebra
from testquery import ClangRunner
from verifier import verify
import raco.myrial.query_tests as query_tests


class CMyrialTests(query_tests.TestQueryFunctions):
    _i = 0

    def create_c_tables(self):
        CMyrialTests.emp_table

    def setUp(self):
        self.db = ClangTestDatabase()
        super(CMyrialTests, self).setUp()

    def check_result(self, query, expected, test_logical=False,
                     skip_json=False, output='OUTPUT', scheme=None):

        plan = self.get_physical_plan(query, CCAlgebra())
        print plan

        # generate code in the target language
        code = ""
        code += compile(plan)

        name = "testquery_%d" % self._i
        self._i+=1

        fname = name+'.cpp'
        with open(fname, 'w') as f:
            f.write(code)

        runner = ClangRunner()
        testoutfn = runner.run(name, "/tmp")

        expectedfn = "expected.txt"
        with open(expectedfn, 'w') as wf:
            writer = csv.writer(wf, delimiter=' ')
            for tup in expected:
                writer.writerow(tup)

        verify(testoutfn, expectedfn, False)

    # tests to skip
    def test_aggregate_with_unbox(self):
        pass

    def test_case_unbox(self):
        pass

    def test_compound_aggregate(self):
        pass

    def test_compound_groupby(self):
        # requires grouping by more then 1, AVG
        pass

    def test_distinct_aggregate_combinations(self):
        pass

    def test_empty_groupby(self):
        pass

    def test_impure_aggregate_colref(self):
        pass

    def test_impure_aggregate_unbox(self):
        pass

    def test_nary_groupby(self):
        # requires grouping by multiple attributes
        pass

    def test_sequence(self):
        # requires Sequence of multiple lines
        pass

    def test_answer_to_everything_function(self):
        # requires SingletonRelation
        pass

    def test_table_literal_scalar_expression(self):
        # requires SingletonRelation
        pass

    def test_avg(self):
        # requires AVG
        pass

    def test_stdev(self):
        # need to override because calls db.execute
        pass


if __name__ == '__main__':
    unittest.main()
