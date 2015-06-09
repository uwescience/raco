import unittest
from testquery import checkquery
from testquery import GrappalangRunner
from generate_test_relations import generate_default
from generate_test_relations import need_generate
from raco.language.grappalang import GrappaAlgebra
import raco.language.grappalang as grappalang
from raco.platform_tests import MyriaLPlatformTestHarness, MyriaLPlatformTests
from raco.compile import compile
from nose.plugins.skip import SkipTest

import sys
sys.path.append('./examples')
from osutils import Chdir
import os

import raco.viz as viz

import logging
logging.basicConfig(level=logging.DEBUG)

def is_skipping():
    return not ('RACO_GRAPPA_TESTS' in os.environ
                and int(os.environ['RACO_GRAPPA_TESTS']) == 1)

def raise_skip_test(query=None):
     if not is_skipping():
        return None

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
            # FIXME: see issue #348; always skipping shuffle tests because it got broken
            raise SkipTest()

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

        #raise Exception()
        raise_skip_test(query)

        with Chdir("c_test_environment") as d:
            checkquery(name, GrappalangRunner(binary_input=False))

    def setUp(self):
        super(MyriaLGrappaTest, self).setUp()
        if not is_skipping():
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
        # test depends on determinism in
        # argmax_uda.sql. To do this we
        # use dataset I3, where column c is unique

        self.check_sub_tables("""
        {UDA}
        I3 = SCAN(%(I3)s);
        out = select a, ArgMax(b, c) from I3;
        STORE(out, OUTPUT);
        """.format(UDA=self._uda_def()), "argmax_uda")

    def test_argmax_all_uda(self):
        # Always skip until no-key UDA is properly supported with decomposable UDA
        raise SkipTest()
        self.check_sub_tables("""
        {UDA}
        I3 = SCAN(%(I3)s);
        out = select ArgMax(b, c) from I3;
        STORE(out, OUTPUT);
        """.format(UDA=self._uda_def()), "argmax_all_uda")
        # TODO only test decomposable argmax here, as the non decomposable no-key is less useful

    def test_two_key_hash_join(self):
        self.check_sub_tables("""
        R3 = SCAN(%(R3)s);
        T3 = SCAN(%(T3)s);
        J = [from R3, T3 where R3.a=T3.a and R3.b=T3.b emit R3.c, T3.c];
        STORE(J, OUTPUT);
        """, "two_key_hash_join")

    def test_two_key_hash_join_swap(self):
        self.check_sub_tables("""
        R3 = SCAN(%(R3)s);
        T3 = SCAN(%(T3)s);
        J = [from R3, T3 where R3.a=T3.b and R3.b=T3.a emit R3.c, T3.c];
        STORE(J, OUTPUT);
        """, "two_key_hash_join_swap")

    def test_three_way_three_key_hash_join(self):
        self.check_sub_tables("""
        R3 = SCAN(%(R3)s);
        T3 = SCAN(%(T3)s);
        S3 = SCAN(%(S3)s);
        J = [from R3, T3, S3 where R3.a=T3.a and R3.b=T3.b and R3.c=T3.c
         and R3.a=S3.a and R3.b=S3.b and R3.c=S3.c emit R3.c, T3.c, S3.c];
        STORE(J, OUTPUT);
        """, "three_way_three_key_hash_join")


if __name__ == '__main__':
    unittest.main()
