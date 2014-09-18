import unittest
from testquery import checkquery
from testquery import GrappalangRunner
from generate_test_relations import generate_default
from generate_test_relations import need_generate
from raco.language.grappalang import GrappaAlgebra
import raco.language.grappalang as grappalang
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

        if kwargs.get('join_type', None) == 'symmetric_hash':
            kwargs['join_type'] = grappalang.GrappaSymmetricHashJoin
        elif kwargs.get('join_type', None) == 'shuffle_hash':
            kwargs['join_type'] = grappalang.GrappaShuffleHashJoin

        kwargs['target_alg'] = GrappaAlgebra()

        plan = self.get_physical_plan(query, **kwargs)
        physical_dot = viz.operator_to_dot(plan)

        with open(os.path.join("c_test_environment",
                               "{gname}.physical.dot".format(gname=gname)), 'w') as dwf:
            dwf.write(physical_dot)

        # generate code in the target language
        code = compile(plan)

        fname = os.path.join("c_test_environment", "{gname}.cpp".format(gname=gname))
        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, 'w') as f:
            f.write(code)

        raise_skip_test(query)

        with Chdir("c_test_environment") as d:
            checkquery(name, GrappalangRunner())

    def setUp(self):
        raise_skip_test()
        super(MyriaLGrappaTest, self).setUp()
        with Chdir("c_test_environment") as d:
            targetpath = os.path.join(os.environ.copy()['GRAPPA_HOME'], 'build/Make+Release/applications/join')
            if need_generate(targetpath):
                generate_default(targetpath)

    def _uda_def(self):
        uda_def_path = os.path.join("c_test_environment", "testqueries", "argmax.myl")
        with open(uda_def_path, 'r') as ro:
            return ro.read()

    # Grappa-only tests
    def test_argmax_uda(self):

        self.check_sub_tables("""
        {UDA}
        R3 = SCAN(%(R3)s);
        out = select a, ArgMax(b, c) from R3;
        STORE(out, OUTPUT);
        """.format(UDA=self._uda_def()), "argmax_uda")

    def test_argmax_all_uda(self):

        self.check_sub_tables("""
        {UDA}
        R3 = SCAN(%(R3)s);
        out = select ArgMax(b, c) from R3;
        STORE(out, OUTPUT);
        """.format(UDA=self._uda_def()), "argmax_all_uda")


if __name__ == '__main__':
    unittest.main()
