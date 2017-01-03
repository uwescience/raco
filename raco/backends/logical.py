import raco.rules as rules
from raco.backends import Algebra


class OptLogicalAlgebra(Algebra):

    @staticmethod
    def opt_rules(**kwargs):
        return [rules.RemoveTrivialSequences(),
                rules.SimpleGroupBy(),
                rules.SplitSelects(),
                rules.PushSelects(),
                rules.MergeSelects(),
                rules.ProjectToDistinctColumnSelect(),
                rules.JoinToProjectingJoin(),
                rules.PushApply(),
                rules.RemoveUnusedColumns(),
                rules.PushApply(),
                rules.RemoveUnusedColumns(),
                rules.PushApply(),
                rules.DeDupBroadcastInputs()]
