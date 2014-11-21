from nose.plugins.skip import SkipTest
import os
from osutils import Chdir
import sys
import unittest

from raco.language.grappalang import GrappaAlgebra

from generate_test_relations import generate_default, need_generate
from platform_tests import DatalogPlatformTest
from testquery import checkquery, checkstore, GrappalangRunner

sys.path.append('./examples')
from emitcode import emitCode


class DatalogGrappaTest(unittest.TestCase, DatalogPlatformTest):
    def check(self, query, name):
        emitCode(query, 'grappa_%s' % name, GrappaAlgebra, dir='c_test_environment')
        # TODO actually be able to check the query
        raise SkipTest(query)
        with Chdir("c_test_environment") as d:
            checkquery(name, GrappalangRunner())

    def check_file(self, query, name):
        # TODO implement this function
        raise SkipTest(query)

    def setUp(self):
        # TODO instead of returning, we should do something with GRAPPA_HOME
        return
        with Chdir("c_test_environment") as d:
            targetpath = os.path.join(os.environ.copy()['GRAPPA_HOME'], 'build/Make+Release/applications/join')
            if need_generate(targetpath):
              generate_default(targetpath)


if __name__ == '__main__':
    unittest.main()
