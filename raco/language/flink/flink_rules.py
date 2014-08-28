import raco.algebra as algebra
from raco.expression import (AttributeRef, UnnamedAttributeRef,
                             NumericLiteral, COUNTALL, COUNT, SUM, MIN, MAX,
                             to_unnamed_recursive, reindex_expr)
import raco.rules as rules


class FlinkAlgebra(object):
    @staticmethod
    def opt_rules():
        pre_rules = [rules.RemoveTrivialSequences(),
                     rules.SimpleGroupBy(),
                     rules.SplitSelects(),
                     rules.PushSelects(),
                     rules.MergeSelects(),
                     rules.ProjectingJoin(),
                     rules.JoinToProjectingJoin()]
        cleanup_rules = [rules.PushApply(),
                         rules.RemoveUnusedColumns(),
                         rules.PushApply(),
                         rules.RemoveUnusedColumns(),
                         rules.PushApply(),
                         rules.RemoveNoOpApply()]
        flink_group_by = [FlinkGroupBy()]
        flink_projecting_join = [FlinkProjectingJoin()]
        return (pre_rules + flink_group_by + cleanup_rules
                + flink_projecting_join + cleanup_rules)


class FlinkGroupBy(rules.Rule):
    """Flink aggregates are required to not change the signature of the input
    or the order of the columns. Thus we prepend an apply that makes this
    happen."""

    def fire(self, op):
        if not isinstance(op, algebra.GroupBy):
            return op

        if (not all(isinstance(e, AttributeRef) for e in op.grouping_list)
            or not all(isinstance(a.input, AttributeRef)
                       for a in op.aggregate_list)):
            raise NotImplementedError("FlinkGroupBy expects a simple groupby")

        emit_exprs = [g for g in op.grouping_list]
        op.grouping_list = [UnnamedAttributeRef(i)
                            for i in range(len(emit_exprs))]

        for i, a in enumerate(op.aggregate_list):
            pos = len(emit_exprs)
            if isinstance(a, (COUNT, COUNTALL)):
                emit_exprs.append(NumericLiteral(1))
                op.aggregate_list[i] = SUM(UnnamedAttributeRef(pos))
            elif isinstance(a, (MAX, MIN)):
                emit_exprs.append(a.input)
                a.input = UnnamedAttributeRef(pos)
            else:
                raise NotImplementedError(
                    "Flink GroupBy with aggregate {a}".format(a=a))

        apply = algebra.Apply(emitters=[(None, e) for e in emit_exprs],
                              input=op.input)
        op.input = apply
        return op


class FlinkProjectingJoin(rules.Rule):
    """Flink cannot emit columns in a different order than they come in.
    Fix this by permuting them on the way in."""

    def fire(self, op):
        if not isinstance(op, algebra.ProjectingJoin):
            return op

        left_scheme = op.left.scheme()
        right_scheme = op.right.scheme()
        left_len = len(left_scheme)
        input_scheme = left_scheme + right_scheme
        output_cols = [c.get_position(input_scheme) for c in op.output_columns]
        sorted_cols = sorted(output_cols)
        if sorted_cols == output_cols:
            return op

        # Swap the correct columns in the left child using an Apply
        left_cols = [c for c in output_cols if c < left_len]
        left_col_map = {c: s for c, s in zip(left_cols, sorted(left_cols))}
        left_refs = [UnnamedAttributeRef(left_col_map.get(i) or i)
                     for i in range(left_len)]
        emitters = [(None, ref) for ref in left_refs]
        left_apply = algebra.Apply(emitters=emitters, input=op.left)

        # Swap the correct columns in the right child using an Apply
        right_cols = [c - left_len for c in output_cols if c >= left_len]
        right_col_map = {c: s for c, s in zip(right_cols, sorted(right_cols))}
        right_refs = [UnnamedAttributeRef(right_col_map.get(i) or i)
                      for i in range(len(right_scheme))]
        emitters = [(None, ref) for ref in right_refs]
        right_apply = algebra.Apply(emitters=emitters, input=op.left)

        # Reindex the join condition and output columns
        join_reindex_map = {c: s for c, s in zip(sorted_cols, output_cols)}
        reindex_expr(op.condition, join_reindex_map)

        op.output_columns = [UnnamedAttributeRef(s) for s in sorted_cols]
        op.left = left_apply
        op.right = right_apply

        return op