

from raco.physprop import *

class CommunicationVisitor(object):
    def visit(self, op):
        method = getattr(self, '__visit_' + op.opname().lower())
        method(op)

    def __visit_scan(self, op):
        # TODO: Get partition info from the catalog
        op.how_partitioned = PARTITION_RANDOM
        op.equivalent_columns = ColumnEquivalenceClassSet(num_cols)
        return op

    def __visit_select(self, op):
        # TODO: pay attention to select condition
        op.how_partitioned = op.input.how_partitioned
        op.equivalent_columns = op.input.equivalent_columns

