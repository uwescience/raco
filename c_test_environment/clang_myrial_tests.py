import unittest
from testquery import checkquery
from testquery import ClangRunner
from generate_test_relations import generate_default
from generate_test_relations import need_generate
from raco.language.clang import CCAlgebra
from platform_tests import MyriaLPlatformTestHarness, MyriaLPlatformTests
from raco.compile import compile

import sys
sys.path.append('./examples')
from osutils import Chdir
import os


class ClangTest(MyriaLPlatformTestHarness, MyriaLPlatformTests):
    def check(self, query, name):
        plan = self.get_physical_plan(query, CCAlgebra())
        print plan

        # generate code in the target language
        code = ""
        code += compile(plan)

        fname = name+'.cpp'

        with Chdir("c_test_environment") as d:
            with open(fname, 'w') as f:
                f.write(code)

            checkquery(name, ClangRunner())
            #os.remove("%s.cpp" % name)

    def setUp(self):
        super(ClangTest, self).setUp()
        with Chdir("c_test_environment") as d:
            if need_generate():
                generate_default()


if __name__ == '__main__':
    unittest.main()
