import unittest

import scheme
import relation_key
from algebra import *
from expression import *
from comm import *
from physprop import *


class CommunicationTests(unittest.TestCase):

    def setUp(self):
        self.cv = CommunicationVisitor()
        self.scheme = scheme.Scheme([("x", "int"), ("y", "int"), ("z", "int"),
                                    ("w", "int")])
        self.rel_key = relation_key.RelationKey.from_string("andrew:adhoc:ZZZ")

    def validate(self, op_in, cevs, how_partitioned):
        def rec(op):
            op.apply(rec)
            return self.cv.visit(op)

        op_out = rec(op_in)
        self.assertEquals(op_out.how_partitioned, how_partitioned)
        self.assertEquals(op_out.column_equivalences, cevs)

    def test_scan(self):
        op_in = Scan(self.rel_key, self.scheme)
        self.validate(op_in, ColumnEquivalenceClassSet(4), PARTITION_RANDOM)

    def test_select(self):
        scan = Scan(self.rel_key, self.scheme)
        cond = GT(UnnamedAttributeRef(1), NumericLiteral(1))
        select = Select(condition=cond, input=scan)

        self.validate(select, ColumnEquivalenceClassSet(4), PARTITION_RANDOM)

    def test_select_equal_columns(self):
        scan = Scan(self.rel_key, self.scheme)
        cond = EQ(UnnamedAttributeRef(2), UnnamedAttributeRef(3))
        select = Select(condition=cond, input=scan)

        cevs = ColumnEquivalenceClassSet(4)
        cevs.merge(2, 3)
        self.validate(select, cevs, PARTITION_RANDOM)
