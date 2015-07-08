from raco.backends.sparql import SPARQLAlgebra
from raco.platform_tests import MyriaLPlatformTestHarness
import raco.compile


class SPARQLTests(object):
    # TODO: refactor MyrialPlatformTests to share code
    def check_sub_tables(self, query, name, **kwargs):
        self.check(query % self.tables, name, **kwargs)

    def test_scan(self):
        self.check_sub_tables("""
        T1 = SCAN(%(T1)s);
        STORE(T1, OUTPUT);
        """, "scan")

    def test_select(self):
        self.check_sub_tables("""
        T1 = SCAN(%(T1)s);
        x = [FROM T1 WHERE a>5 EMIT a];
        STORE(x, OUTPUT);
        """, "select")

    def test_join(self):
        self.check_sub_tables("""
        T3 = SCAN(%(T3)s);
        R3 = SCAN(%(R3)s);
        out = JOIN(T3, b, R3, b);
        out2 = [FROM out WHERE $3 = $5 EMIT $0, $3];
        STORE(out2, OUTPUT);
        """, "join")


class SPARQLMyriaLTests(MyriaLPlatformTestHarness, SPARQLTests):

    def check(self, query, name):
        plan = self.get_physical_plan(query, target_alg=SPARQLAlgebra())

        sparql = raco.compile.compile(plan)

        # TODO pretty lenient tests: is it a non empty string?
        assert isinstance(sparql, ''.__class__)
        assert sparql != ''
