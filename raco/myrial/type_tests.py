"""Various tests of type safety."""
import unittest

from raco.fakedb import FakeDatabase
from raco.scheme import Scheme
from raco.myrial.myrial_test import MyrialTestCase
from collections import Counter


class TypeTests(MyrialTestCase):
    schema = Scheme(
        [("clong", "LONG_TYPE"),
         ("cint", "INT_TYPE"),
         ("cstring", "STRING_TYPE"),
         ("cfloat", "DOUBLE_TYPE")])

    def setUp(self):
        super(TypeTests, self).setUp()
        self.db.ingest("public:adhoc:mytable", Counter(), TypeTests.schema)

    def noop_test(self):
        query = """
        X = SCAN(public:adhoc:mytable);
        STORE(X, OUTPUT);
        """

        self.check_scheme(query, TypeTests.schema)
