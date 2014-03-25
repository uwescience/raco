import unittest

import scheme
import relation_key
from algebra import *
from comm import *
from physprop import *


class CommunicationTests(unittest.TestCase):

    def setUp(self):
        self.cv = CommunicationVisitor()
        self.scheme = scheme.Scheme([("x", "int"), ("y", "int"), ("z", "int"),
                                    ("w", "int")])
        self.rel_key = relation_key.RelationKey.from_string("andrew:adhoc:ZZZ")

    def __validate(self, op_out, cevs, how_partitioned):
        self.assertEquals(op_out.how_partitioned, how_partitioned)
        self.assertEquals(op_out.column_equivalences, cevs)

    def test_scan(self):
        op_in = Scan(self.rel_key, self.scheme)
        op_out = self.cv.visit(op_in)

        self.__validate(op_out, ColumnEquivalenceClassSet(4), PARTITION_RANDOM)
