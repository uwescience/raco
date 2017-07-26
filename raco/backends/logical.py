import raco.rules as rules
from raco.backends import Algebra


class OptLogicalAlgebra(Algebra):

    @staticmethod
    def opt_rules(**kwargs):
        return [rules.remove_trivial_sequences,
                rules.simple_group_by,
                rules.push_select,
                rules.push_project,
                rules.push_apply,
                [rules.DeDupBroadcastInputs()]]
