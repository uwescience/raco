"""Various tests of type safety."""
import unittest

from raco.fakedb import FakeDatabase
from raco.scheme import Scheme
from raco.myrial.myrial_test import MyrialTestCase
from raco.expression import TypeSafetyViolation
from collections import Counter


class TypeTests(MyrialTestCase):
    schema = Scheme(
        [("clong", "LONG_TYPE"),
         ("cint", "INT_TYPE"),
         ("cstring", "STRING_TYPE"),
         ("cfloat", "DOUBLE_TYPE"),
         ("cbool", "BOOLEAN_TYPE"),
         ("cdate", "DATETIME_TYPE")])

    def setUp(self):
        super(TypeTests, self).setUp()
        self.db.ingest("public:adhoc:mytable", Counter(), TypeTests.schema)

    def test_noop(self):
        query = """
        X = SCAN(public:adhoc:mytable);
        STORE(X, OUTPUT);
        """

        self.check_scheme(query, TypeTests.schema)

    def test_invalid_eq1(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT clong=cstring];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_eq2(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT cfloat=cdate];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_ge(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT cfloat>=cstring];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_lt(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT cfloat<cdate];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_and(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT cfloat AND cdate];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_or(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT cfloat OR cdate];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_not(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT not cdate];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_plus(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT cdate + cint];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_times(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT cdate * cstring];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_divide(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT cdate / clong];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_divide(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT cfloat / cfloat AS y];
        STORE(X, OUTPUT);
        """
        schema = Scheme([('y', "DOUBLE_TYPE")])
        self.check_scheme(query, schema)

    def test_idivide(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT cfloat // cint AS y];
        STORE(X, OUTPUT);
        """
        schema = Scheme([('y', "LONG_TYPE")])
        self.check_scheme(query, schema)

    def test_invalid_case1(self):
        query = """
        t = SCAN(public:adhoc:mytable);
        rich = [FROM t EMIT
                CASE WHEN clong <= 5000 THEN "poor"
                     WHEN clong <= 25000 THEN 5.5
                     ELSE "rich" END];
        STORE(rich, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_case2(self):
        query = """
        t = SCAN(public:adhoc:mytable);
        rich = [FROM t EMIT
                CASE WHEN clong <= 5000 THEN "poor"
                     WHEN clong <= 25000 THEN "middle class"
                     ELSE 1922 END];
        STORE(rich, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_cos(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS Y
             WHERE cos(cbool) > 1.0 EMIT *];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_tan(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS Y
             WHERE tan(cstring) > 1.0 EMIT *];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_pow(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT POW(cfloat, cstring)];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_substr(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT SUBSTR(0, 3, cfloat)];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)

    def test_invalid_len(self):
        query = """
        X = [FROM SCAN(public:adhoc:mytable) AS X EMIT LEN(cfloat)];
        STORE(X, OUTPUT);
        """
        with self.assertRaises(TypeSafetyViolation):
            self.check_scheme(query, None)
