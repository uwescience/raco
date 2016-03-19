import unittest
from testquery import checkquery, checkstore
from testquery import GrappalangRunner
from generate_test_relations import generate_default
from generate_test_relations import need_generate
from raco.backends.radish import GrappaAlgebra
import raco.backends.radish as grappalang
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
    def check(self, query, name, join_type=None, emit_print='console', **kwargs):
        gname = "grappa_{name}".format(name=name)

        if join_type is None:
            pass
        elif join_type == 'symmetric_hash':
            kwargs['join_type'] = grappalang.GrappaSymmetricHashJoin
        elif join_type == 'shuffle_hash':
            kwargs['join_type'] = grappalang.GrappaShuffleHashJoin
            # FIXME: see issue #348; always skipping shuffle tests because it got broken
            raise SkipTest(query)
        else:
            raise NotImplementedError(
                "join_type {} not supported".format(join_type))

        kwargs['target_alg'] = GrappaAlgebra(emit_print=emit_print)

        plan = self.get_physical_plan(query, **kwargs)
        physical_dot = viz.operator_to_dot(plan)

        with open(os.path.join("c_test_environment",
                               "{gname}.physical.dot".format(gname=gname)), 'w') as dwf:
            dwf.write(physical_dot)

        # generate code in the target language
        # test_mode=True turns on extra checks like assign-once instance
        #    variables for operators
        code = compile(plan, test_mode=True)

        fname = os.path.join("c_test_environment", "{gname}.cpp".format(gname=gname))
        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, 'w') as f:
            f.write(code)

        #raise Exception()

        raise_skip_test(query)

        with Chdir("c_test_environment") as d:
            if emit_print == 'file':
                checkstore(name, GrappalangRunner(binary_input=False))
            else:
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
        with self.assertRaises(NotImplementedError):
            self.check_sub_tables("""
            {UDA}
            I3 = SCAN(%(I3)s);
            out = select ArgMax(b, c) from I3;
            STORE(out, OUTPUT);
            """.format(UDA=self._uda_def()), "argmax_all_uda")
            # TODO only test decomposable argmax here, as the non decomposable no-key is less useful

    def test_builtin_and_UDA(self):
        self.check_sub_tables("""
        {UDA}
        I3 = SCAN(%(I3)s);
        out = select a, ArgMax(b, c), SUM(b) from I3;
        STORE(out, OUTPUT);
        """.format(UDA=self._uda_def()), "builtin_and_UDA")

    def test_multi_builtin(self):
        self.check_sub_tables("""
        I3 = SCAN(%(I3)s);
        out = select c, MAX(a), SUM(b) from I3;
        STORE(out, OUTPUT);
        """, "multi_builtin")


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

    def test_symmetric_array_repr(self):
        q = self.myrial_from_sql(['T1'], "select")
        self.check(q, "select", scan_array_repr='symmetric_array')

    def test_indexed_strings(self):
        q = self.myrial_from_sql(["C3", "C3"], "join_string_key")
        self.check(q, "join_string_key", external_indexing=True)

    def test_shuffle_hash_join(self):
        """
        GrappaShuffleHashJoin is outdated.
        For example, it only supports single attribute key.
        """
        self.check_sub_tables("""
        T3 = SCAN(%(T3)s);
        R3 = SCAN(%(R3)s);
        out = JOIN(T3, b, R3, b);
        out2 = [FROM out WHERE $3 = $5 EMIT $0, $3];
        STORE(out2, OUTPUT);
        """, "join", join_type='shuffle_hash')

    def test_while(self):
        """
        Test a minimal while loop
        """
        self.check("""
            i = [4];
            do
                i = [from i emit *i - 1];
            while [from i where *i > 0 emit *i];
            store(i, OUTPUT);
        """, "while")

    def test_while_union_all(self):
        """
        Test UNIONALL into StoreTemp in a While loop
        """
        self.check("""
            m = [1234];
            do
                m = UNIONALL(m, m);
                cnt = select count($0) as c from m;
                eqfive = select case
                                when c = 8 then 0
                                else 1
                                 end
                        from cnt;
            while [from eqfive where *eqfive emit *eqfive];
            store(m, OUTPUT);
        """, "while_union_all")

    def _while_join_query(self):
        return """
            s = scan(%(T3)s);
            i = [2];
            do
                i = [from i emit *i - 1];
                s = select s1.b, s1.c, s1.a from s s1, s s2 where s1.a=s2.b;
            while [from i where *i > 0 emit *i];
            store(s, OUTPUT);
        """

    def test_while_repeat_hash_join(self):
        self.check_sub_tables(self._while_join_query(), "while_repeat_join")

    def test_while_repeat_sym_hash_join(self):
        self.check_sub_tables(self._while_join_query(), "while_repeat_join", join_type='symmetric_hash' )
    
    def test_while_repeat_groupby(self):
        self.check_sub_tables("""
            s = scan(%(T3)s);
            i = [2];
            do
                i = [from i emit *i - 1];
                s = select SUM(s.a) as a,
                    s.c as b,
                    SUM(s.b) as c from s;
            while [from i where *i > 0 emit *i];
            store(s, OUTPUT);
        """, "while_repeat_groupby")


if __name__ == '__main__':
    unittest.main()
