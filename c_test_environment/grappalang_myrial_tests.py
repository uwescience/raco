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


def raise_skip_test(query=None):
  if 'RACO_GRAPPA_TESTS' in os.environ:
    if int(os.environ['RACO_GRAPPA_TESTS']) == 1:
        return

  if query is not None:
    raise SkipTest(query)
  else:
    raise SkipTest()


class MyriaLGrappaTest(MyriaLPlatformTestHarness, MyriaLPlatformTests):
    def check(self, query, name, **kwargs):
        gname = "grappa_{name}".format(name=name)

        kwargs['target_alg'] = GrappaAlgebra()
        plan = self.get_physical_plan(query, **kwargs)
        physical_dot = viz.operator_to_dot(plan)

        with Chdir("c_test_environment") as d:
            with open("{gname}.physical.dot".format(gname=gname), 'w') as dwf:
                dwf.write(physical_dot)

            # generate code in the target language
            code = compile(plan)

            fname = "{gname}.cpp".format(gname=gname)
            if os.path.exists(fname):
                os.remove(fname)
            with open(fname, 'w') as f:
                f.write(code)
        
            raise_skip_test(query)
            checkquery(name, GrappalangRunner())

    def setUp(self):
        raise_skip_test()
        super(MyriaLGrappaTest, self).setUp()
        with Chdir("c_test_environment") as d:
            targetpath = os.path.join(os.environ.copy()['GRAPPA_HOME'], 'build/Make+Release/applications/join')
            if need_generate(targetpath):
                generate_default(targetpath)


if __name__ == '__main__':
    unittest.main()
