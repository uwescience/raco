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

    def visit_apply(self, op):
        new_pos = {}  # map from original column position to first output pos
        exprs = {}  # map from expression to output position set

        for i, emitter in enumerate(op.emitters):
            s = exprs.get(emitter, set())
            s.add(i)

            assert not isinstance(emitter, expression.NamedAttributeRef)
            if (isinstance(emitter, expression.UnnamedAttributeRef) and
                not emitter.position in new_pos):  # noqa
                new_pos[emitter.position] = i

        cevs_in = op.input.column_equivalences
        cevs_out = ColumnEquivalenceClassSet(len(op.scheme()))

        # Merge input column equivalences that are preserved
        for pos_set in cevs_in:
            retained = [new_pos[x] for x in pos_set if x in new_pos]
            cevs_out.merge_set(retained)

        # Merge any output columns that have a common expression
        for pos_set in exprs.itervalues():
            cevs_out.merge_set(pos_set)

        op.column_equivalences = cevs_out

        # The output maintains the input partition if all columns are preserved
        # in the same order
        op.how_partitioned = PARTITION_RANDOM
        hp_in = op.input.how_partitioned

        if isinstance(hp_in, set):
            hp_out = [new_pos[x] for x in hp_in if x in new_pos]
            if len(hp_out) == len(hp_in) and is_sorted(hp_out):
                op.how_partitioned = set(hp_out)
                assert len(op.how_partitioned) == len(hp_in)

        return op
