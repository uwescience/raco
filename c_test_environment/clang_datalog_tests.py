import unittest
from testquery import checkquery, checkstore
from testquery import ClangRunner
from generate_test_relations import generate_default
from generate_test_relations import need_generate
import raco.backends.cpp as clang
import raco.backends.cpp.cppcommon as cppcommon
from raco.platform_tests import DatalogPlatformTest

import sys
sys.path.append('./examples')
from osutils import Chdir
from raco.cpp_datalog_utils import emitCode
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
            emitCode(query, name, clang.CCAlgebra, emit_print=cppcommon.EMIT_FILE)
            checkstore(name, ClangRunner())

    def setUp(self):
        with Chdir("c_test_environment") as d:
            if need_generate():
                generate_default()


if __name__ == '__main__':
    unittest.main()
