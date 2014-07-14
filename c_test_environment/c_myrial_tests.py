import unittest

from raco.compile import compile
from raco.language.clang import CCAlgebra
from testquery import ClangRunner
from verifier import verify
import raco.myrial.query_tests as query_tests


class CMyrialTests(query_tests.TestQueryFunctions):
    def check_result(self, query, expected, test_logical=False,
                     skip_json=False, output='OUTPUT', scheme=None):

        plan = self.get_physical_plan(query, CCAlgebra())

        # generate code in the target language
        code = ""
        code += compile(plan)

        name = "testquery"

        fname = name+'.cpp'
        with open(fname, 'w') as f:
            f.write(code)

        runner = ClangRunner()
        testoutfn = runner.run(name, "/tmp")

        expectedfn = "expected.txt"
        with open("expected.txt", 'w') as wf:
            wf.write(expected)

        verify(testoutfn, expectedfn, False)

    # tests to skip
    def test_aggregate_with_unbox(self):
        pass

    def test_case_unbox(self):
        pass

    def test_compound_aggregate(self):
        pass

    def test_compound_groupby(self):
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
        pass

    def test_sequence(self):
        pass




if __name__ == '__main__':
    unittest.main()
