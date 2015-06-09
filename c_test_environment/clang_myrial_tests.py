import unittest
from testquery import checkquery
from testquery import ClangRunner
from generate_test_relations import generate_default
from generate_test_relations import need_generate
from raco.language.clang import CCAlgebra
from raco.platform_tests import MyriaLPlatformTestHarness, MyriaLPlatformTests
from raco.compile import compile

import sys
sys.path.append('./examples')
from osutils import Chdir
import os

import raco.viz as viz

#import logging
#logging.basicConfig(level=logging.DEBUG)


class MyriaLClangTest(MyriaLPlatformTestHarness, MyriaLPlatformTests):
    def check(self, query, name, **kwargs):
        kwargs['target_alg'] = CCAlgebra()
        plan = self.get_physical_plan(query, **kwargs)
        physical_dot = viz.operator_to_dot(plan)
        with open(os.path.join("c_test_environment", "%s.physical.dot"%(name)), 'w') as dwf:
            dwf.write(physical_dot)

        # generate code in the target language
        code = compile(plan)

        fname = os.path.join("c_test_environment", "{name}.cpp".format(name=name))
        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, 'w') as f:
            f.write(code)

        with Chdir("c_test_environment") as d:
            checkquery(name, ClangRunner())

    def setUp(self):
        super(MyriaLClangTest, self).setUp()
        with Chdir("c_test_environment") as d:
            if need_generate():
                generate_default()


if __name__ == '__main__':
    unittest.main()
