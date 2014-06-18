import unittest
from testquery import checkquery
from testquery import GrappalangRunner
from generate_test_relations import generate_default
from generate_test_relations import need_generate
from raco.language import GrappaAlgebra
from platform_tests import PlatformTest

import sys
import os
sys.path.append('./examples')
from emitcode import emitCode
from osutils import Chdir


class GrappaTest(unittest.TestCase, PlatformTest):
    def check(self, query, name):
        with Chdir("c_test_environment") as d:
            emitCode(query, 'grappa_%s' % name, GrappaAlgebra)
            checkquery(name, GrappalangRunner())

    def setUp(self):
        with Chdir("c_test_environment") as d:
            targetpath = os.path.join(os.environ.copy()['GRAPPA_HOME'], 'build/Make+Release/applications/join')
            if need_generate(targetpath):
              generate_default(targetpath)


if __name__ == '__main__':
    unittest.main()
