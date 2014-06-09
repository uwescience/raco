import rules


class ReplaceWithView(rules.Rule):
    def fire(self, expr):
        return expr

    def __str__(self):
        return "Replace with materialized view"
