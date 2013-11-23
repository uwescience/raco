from raco import algebra
from raco import expression

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
            return algebra.Join(expression.EQ(expression.NumericLiteral(1), expression.NumericLiteral(1)), expr.left, expr.right)
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
                return algebra.ProjectingJoin(expr.input.condition, expr.input.left, expr.input.right, expr.columnlist)
        return expr

    def __str__(self):
        return "Project, Join => ProjectingJoin"

class JoinToProjectingJoin(Rule):
    """A rewrite rule for turning every Join into a ProjectingJoin"""
    def fire(self, expr):
        if not isinstance(expr, algebra.Join) or isinstance(expr,
                algebra.ProjectingJoin):
            return expr

        return algebra.ProjectingJoin(expr.condition, expr.left, expr.right, expr.scheme().ascolumnlist())

    def __str__(self):
        return "Join => ProjectingJoin"
