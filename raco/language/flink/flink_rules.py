import raco.algebra as algebra
from raco.expression import (AttributeRef, UnnamedAttributeRef,
                             NumericLiteral, COUNTALL, COUNT, SUM, MIN, MAX)
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
        flink_rules = [FlinkGroupBy()]
        post_rules = [rules.PushApply(),
                      rules.RemoveUnusedColumns(),
                      rules.PushApply(),
                      rules.RemoveUnusedColumns(),
                      rules.PushApply(),
                      rules.RemoveNoOpApply()]
        return pre_rules + flink_rules + post_rules


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