

from raco.physprop import *


class CommunicationVisitor(object):

    def visit(self, op):
        return getattr(self, 'visit_' + op.opname().lower())(op)

    def visit_scan(self, op):
        # TODO: Get partition info from the catalog
        op.how_partitioned = PARTITION_RANDOM
        op.column_equivalences = ColumnEquivalenceClassSet(len(op.scheme()))
        return op

    def visit_select(self, op):
        op.how_partitioned = op.input.how_partitioned
        op.column_equivalences = op.input.column_equivalences

        return op
