import unittest
from testquery import checkquery
from testquery import testdbname
from testquery import GrappalangRunner
from generate_test_relations import generate_default
from raco.language import GrappaAlgebra
from platform_tests import PlatformTest

import sys
import os
sys.path.append('./examples')
from emitcode import emitCode
from osutils import Chdir


class GrappaTest(unittest.TestCase, PlatformTest):
    def check(self, query, name):
        chdir = Chdir("c_test_environment")
        emitCode(query, name, GrappaAlgebra)
        checkquery(name, GrappalangRunner())

    def setUp(self):
        chdir = Chdir("c_test_environment")
        if not os.path.isfile(testdbname()):
            generate_default()


if __name__ == '__main__':
    unittest.main()
