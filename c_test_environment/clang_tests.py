import unittest
from testquery import checkquery
from testquery import ClangRunner
from generate_test_relations import generate_default
from generate_test_relations import need_generate
from raco.language.clang import CCAlgebra
from platform_tests import PlatformTest

import sys
sys.path.append('./examples')
from emitcode import emitCode
from osutils import Chdir


class ClangTest(unittest.TestCase, PlatformTest):
    def check(self, query, name):
        with Chdir("c_test_environment") as d:
            emitCode(query, name, CCAlgebra)
            checkquery(name, ClangRunner())

    def setUp(self):
        with Chdir("c_test_environment") as d:
          if need_generate():
                generate_default()


if __name__ == '__main__':
    unittest.main()
