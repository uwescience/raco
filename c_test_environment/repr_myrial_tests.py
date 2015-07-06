import unittest
from generate_test_relations import generate_default
from generate_test_relations import need_generate
from raco.backends.myria import MyriaLeftDeepTreeAlgebra
from raco.platform_tests import MyriaLPlatformTestHarness, MyriaLPlatformTests
from raco.from_repr import plan_from_repr

import sys
sys.path.append('./examples')
from osutils import Chdir


class MyriaLReprTest(MyriaLPlatformTestHarness, MyriaLPlatformTests):
    def check(self, query, name, **kwargs):
        kwargs['target_alg'] = MyriaLeftDeepTreeAlgebra()
        plan = self.get_physical_plan(query, **kwargs)
        assert plan == plan_from_repr(repr(plan))

    def setUp(self):
        super(MyriaLReprTest, self).setUp()
        with Chdir("c_test_environment") as d:
            if need_generate():
                generate_default()


if __name__ == '__main__':
    unittest.main()
