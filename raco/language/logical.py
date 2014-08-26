import raco.rules as rules


class LogicalAlgebra(object):
    @staticmethod
    def opt_rules():
        return []


class OptLogicalAlgebra(object):
    @staticmethod
    def opt_rules():
        return [rules.RemoveTrivialSequences(),
                rules.SimpleGroupBy(),
                rules.SplitSelects(),
                rules.PushSelects(),
                rules.MergeSelects(),
                rules.ProjectingJoin(),
                rules.JoinToProjectingJoin(),
                rules.PushApply(),
                rules.RemoveUnusedColumns(),
                rules.PushApply(),
                rules.RemoveUnusedColumns(),
                rules.PushApply(),
                rules.RemoveNoOpApply()]