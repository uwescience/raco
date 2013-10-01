
import collections
import math
import unittest

import raco.fakedb
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
import raco.scheme as scheme
import raco.myrial.groupby
import raco.myrial.unpack_from
import raco.myrial.myrial_test as myrial_test

class SetopTestFunctions(myrial_test.MyrialTestCase):

    emp_table1 = collections.Counter([
        (1, 2, "Bill Howe", 25000),
        (1, 2, "Bill Howe", 25000),
        (2,1,"Dan Halperin",90000),
        (3,1,"Andrew Whitaker",5000),
        (3,1,"Andrew Whitaker",5000),
        (4,2,"Shumo Chu",5000),
        (5,1,"Victor Almeida",25000),
        (6,3,"Dan Suciu",90000),
        (7,1,"Magdalena Balazinska",25000)])

    emp_key1 = "andrew:adhoc:employee1"

    emp_table2 = collections.Counter([
        (1, 2, "Bill Howe", 25000),
        (7,1,"Magdalena Balazinska",25000),
        (7,1,"Magdalena Balazinska",25000),
        (8, 2, "JingJing Wang", 47000)])

    emp_key2 = "andrew:adhoc:employee2"

    emp_schema = scheme.Scheme([("id", "int"),
                                ("dept_id", "int"),
                                ("name", "string"),
                                ("salary", "int")])

    def setUp(self):
        super(SetopTestFunctions, self).setUp()

        self.db.ingest(SetopTestFunctions.emp_key1,
                       SetopTestFunctions.emp_table1,
                       SetopTestFunctions.emp_schema)

        self.db.ingest(SetopTestFunctions.emp_key2,
                       SetopTestFunctions.emp_table2,
                       SetopTestFunctions.emp_schema)

    def test_unionall(self):
        query = """
        out = UNIONALL(SCAN(%s), SCAN(%s));
        DUMP(out);
        """ % (self.emp_key1, self.emp_key2)

        expected = self.emp_table1 + self.emp_table2
        self.run_test(query, expected)
