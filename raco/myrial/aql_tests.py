"""Tests of AQL escaping."""

from raco.myrial.myrial_test import MyrialTestCase
from raco.algebra import *
from collections import Counter
from raco.scheme import Scheme


class AqlTests(MyrialTestCase):
    def setUp(self):
        super(AqlTests, self).setUp()
        self.db.ingest("public:adhoc:employees", Counter, Scheme())

    def test_escaped_aql(self):
        aql = 'SELECT * FROM TestArray'
        query = "%%aql %s;" % aql
        plan = self.get_plan(query, logical=True)
        expected = Sequence([Exec(aql, "AQL")])

        self.assertEquals(plan, expected)

    def test_escaped_aql_with_myrial(self):
        aql = 'SELECT * FROM TestArray'
        query = """
        %%aql %s;
        X = SCAN(public:adhoc:employees);
        STORE(X, OUTPUT);""" % aql

        plan = self.get_plan(query, logical=True)

        self.assertIsInstance(plan, Sequence)
        self.assertIsInstance(plan.args[0], Exec)
        self.assertIsInstance(plan.args[1], Store)
