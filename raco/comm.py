import copy

from raco.physprop import *
import expression


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
        op.column_equivalences = copy.copy(op.input.column_equivalences)

        # Add additional equivalences inferred from select condition
        conjuncs = expression.extract_conjuncs(op.condition)
        assert conjuncs  # Must be at least 1

        for conjunc in conjuncs:
            et = expression.is_column_equality_comparison(conjunc)
            if et:
                op.column_equivalences.merge(*et)
        return op
