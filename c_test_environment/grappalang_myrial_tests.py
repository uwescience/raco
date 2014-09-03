import unittest
from testquery import checkquery
from testquery import GrappalangRunner
from generate_test_relations import generate_default
from generate_test_relations import need_generate
from raco.language.grappalang import GrappaAlgebra
from platform_tests import MyriaLPlatformTestHarness, MyriaLPlatformTests
from raco.compile import compile
from nose.plugins.skip import SkipTest

import sys
sys.path.append('./examples')
from osutils import Chdir
import os

import raco.viz as viz

import logging
logging.basicConfig(level=logging.DEBUG)


class MyriaLGrappaTest(MyriaLPlatformTestHarness, MyriaLPlatformTests):
    def check(self, query, name):
        gname = "grappa_{name}".format(name=name)

        plan = self.get_physical_plan(query, GrappaAlgebra())
        physical_dot = viz.operator_to_dot(plan)
        with open("{gname}.physical.dot".format(gname=gname), 'w') as dwf:
            dwf.write(physical_dot)

        # generate code in the target language
        code = compile(plan)

        with Chdir("c_test_environment") as d:
            fname = "{gname}.cpp".format(gname=gname)
            if os.path.exists(fname):
                os.remove(fname)
            with open(fname, 'w') as f:
                f.write(code)
        
            raise SkipTest(query)
            checkquery(name, GrappalangRunner())

    def setUp(self):
        raise SkipTest()
        super(MyriaLGrappaTest, self).setUp()
        with Chdir("c_test_environment") as d:
            targetpath = os.path.join(os.environ.copy()['GRAPPA_HOME'], 'build/Make+Release/applications/join')
            if need_generate(targetpath):
                generate_default(targetpath)


if __name__ == '__main__':
    unittest.main()
