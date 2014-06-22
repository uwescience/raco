from raco import algebra
from raco import expression
from expression import UnnamedAttributeRef

from abc import ABCMeta, abstractmethod


class Rule(object):
    """Argument is an expression tree

    Returns a possibly modified expression tree"""

    __metaclass__ = ABCMeta

    def __call__(self, expr):
        return self.fire(expr)

    @abstractmethod
    def fire(self, expr):
        """Apply this rule to the supplied expression tree"""


class CrossProduct2Join(Rule):
    """A rewrite rule for removing Cross Product"""
    def fire(self, expr):
        if isinstance(expr, algebra.CrossProduct):
            return algebra.Join(expression.EQ(expression.NumericLiteral(1),
                                expression.NumericLiteral(1)),
                                expr.left, expr.right)
        return expr

    def __str__(self):
        return "CrossProduct(left, right) => Join(1=1, left, right)"


class removeProject(Rule):
    """A rewrite rule for removing Projections"""
    def fire(self, expr):
        if isinstance(expr, algebra.Project):
            return expr.input
        return expr

    def __str__(self):
        return "Project => ()"


class OneToOne(Rule):
    def __init__(self, opfrom, opto):
        self.opfrom = opfrom
        self.opto = opto

    def fire(self, expr):
        if isinstance(expr, self.opfrom):
            newop = self.opto()
            newop.copy(expr)
            return newop
        return expr

    def __str__(self):
        return "%s => %s" % (self.opfrom.__name__, self.opto.__name__)


class ProjectingJoin(Rule):
    """A rewrite rule for combining Project after Join into ProjectingJoin"""
    def fire(self, expr):
        if isinstance(expr, algebra.Project):
            if isinstance(expr.input, algebra.Join):
                return algebra.ProjectingJoin(expr.input.condition,
                                              expr.input.left,
                                              expr.input.right,
                                              expr.columnlist)
        return expr

    def __str__(self):
        return "Project, Join => ProjectingJoin"


class JoinToProjectingJoin(Rule):
    """A rewrite rule for turning every Join into a ProjectingJoin"""

    def fire(self, expr):
        if not isinstance(expr, algebra.Join) or \
                isinstance(expr, algebra.ProjectingJoin):
            return expr

        return algebra.ProjectingJoin(expr.condition,
                                      expr.left, expr.right,
                                      expr.scheme().ascolumnlist())

    def __str__(self):
        return "Join => ProjectingJoin"


class SimpleGroupBy(Rule):
    # A "Simple" GroupBy is one that has only AttributeRefs as its grouping
    # fields, and only AggregateExpression(AttributeRef) as its aggregate
    # fields.
    #
    # Even AggregateExpression(Literal) is more complicated than Myria's
    # GroupBy wants to handle. Thus we will insert Apply before a GroupBy to
    # take all the "Complex" expressions away.

    def fire(self, expr):
        if not isinstance(expr, algebra.GroupBy):
            return expr

        child_scheme = expr.input.scheme()

        # A simple grouping expression is an AttributeRef
        def is_simple_grp_expr(grp):
            return isinstance(grp, expression.AttributeRef)

        complex_grp_exprs = [(i, grp)
                             for (i, grp) in enumerate(expr.grouping_list)
                             if not is_simple_grp_expr(grp)]

        # A simple aggregate expression is an aggregate whose input is an
        # AttributeRef
        def is_simple_agg_expr(agg):
            return (isinstance(agg, expression.COUNTALL) or
                    (isinstance(agg, expression.UnaryOperator) and
                     isinstance(agg, expression.AggregateExpression) and
                     isinstance(agg.input, expression.AttributeRef)))

        complex_agg_exprs = [agg for agg in expr.aggregate_list
                             if not is_simple_agg_expr(agg)]

        # There are no complicated expressions, we're okay with the existing
        # GroupBy.
        if not complex_grp_exprs and not complex_agg_exprs:
            return expr

        # Construct the Apply we're going to stick before the GroupBy

        # First: copy every column from the input verbatim
        mappings = [(None, UnnamedAttributeRef(i))
                    for i in range(len(child_scheme))]

        # Next: move the complex grouping expressions into the Apply, replace
        # with simple refs
        for i, grp_expr in complex_grp_exprs:
            mappings.append((None, grp_expr))
            expr.grouping_list[i] = UnnamedAttributeRef(len(mappings) - 1)

        # Finally: move the complex aggregate expressions into the Apply,
        # replace with simple refs
        for agg_expr in complex_agg_exprs:
            mappings.append((None, agg_expr.input))
            agg_expr.input = UnnamedAttributeRef(len(mappings) - 1)

        # Construct and prepend the new Apply
        new_apply = algebra.Apply(mappings, expr.input)
        expr.input = new_apply

        # Don't overwrite expr.grouping_list or expr.aggregate_list, instead we
        # are mutating the objects it contains when we modify grp_expr or
        # agg_expr in the above for loops.
        return expr
