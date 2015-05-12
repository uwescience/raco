import unittest
from testquery import checkquery, checkstore
from testquery import ClangRunner
from generate_test_relations import generate_default
from generate_test_relations import need_generate
import raco.language.clang as clang
import raco.language.clangcommon as clangcommon
from raco.platform_tests import DatalogPlatformTest

import sys
sys.path.append('./examples')
from emitcode import emitCode
from osutils import Chdir
import os


class DatalogClangTest(unittest.TestCase, DatalogPlatformTest):
    def check(self, query, name):
        with Chdir("c_test_environment") as d:
            os.remove("%s.cpp" % name) if os.path.exists("%s.cpp" % name) else None
            emitCode(query, name, clang.CCAlgebra)
            checkquery(name, ClangRunner())

    def check_file(self, query, name):
        with Chdir("c_test_environment") as d:
            os.remove("%s.cpp" % name) if os.path.exists("%s.cpp" % name) else None
            emitCode(query, name, clang.CCAlgebra, emit_print=clangcommon.EMIT_FILE)
            checkstore(name, ClangRunner())

    def setUp(self):
        with Chdir("c_test_environment") as d:
            if need_generate():
                generate_default()


if __name__ == '__main__':
    unittest.main()
