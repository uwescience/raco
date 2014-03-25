import unittest

from mock import patch

import scheme
import relation_key
from algebra import *
from expression import *
from comm import *
from physprop import *


def mock_scan_gen(how_partitioned, cevs):
    def mock_scan(self, op):
        op.how_partitioned = how_partitioned
        op.column_equivalences = cevs
        return op
    return mock_scan


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

    def apply_plan(self):
        scan = Scan(self.rel_key, self.scheme)
        emitters = [('a', UnnamedAttributeRef(3)),
                    ('b', UnnamedAttributeRef(2)),
                    ('c', UnnamedAttributeRef(2))]
        return Apply(emitters=emitters, input=scan)

    def test_apply_duplicate_columns(self):
        """Test that we detect duplicate columns as equivalent."""
        cevs = ColumnEquivalenceClassSet(3)
        cevs.merge(1, 2)
        self.validate(self.apply_plan(), cevs, PARTITION_RANDOM)

    def test_apply_duplicate_columns_preserving(self):
        """Test that column equivalences are preserved across apply."""
        # Mock the scan operator to return partitioned data
        cev_in = ColumnEquivalenceClassSet(4)
        cev_in.merge(3, 2)

        with patch.object(CommunicationVisitor, "visit_scan", mock_scan_gen(
            PARTITION_RANDOM, cev_in)):  # noqa

            cev_out = ColumnEquivalenceClassSet(3)
            cev_out.merge_set([0, 1, 2])
            self.validate(self.apply_plan(), cev_out, PARTITION_RANDOM)
