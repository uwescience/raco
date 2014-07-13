from raco import algebra
from raco import expression
from expression import (accessed_columns, UnnamedAttributeRef,
                        to_unnamed_recursive, is_column_equality_comparison)

from abc import ABCMeta, abstractmethod
import copy
import itertools


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


class RemoveTrivialSequences(Rule):
    def fire(self, expr):
        if not isinstance(expr, algebra.Sequence):
            return expr

        if len(expr.args) == 1:
            return expr.args[0]
        else:
            return expr


class SplitSelects(Rule):
    """Replace AND clauses with multiple consecutive selects."""

    def fire(self, op):
        if not isinstance(op, algebra.Select):
            return op

        conjuncs = expression.extract_conjuncs(op.condition)
        assert conjuncs  # Must be at least 1

        # Normalize named references to integer indexes
        scheme = op.scheme()
        conjuncs = [to_unnamed_recursive(c, scheme)
                    for c in conjuncs]

        op.condition = conjuncs[0]
        op.has_been_pushed = False
        for conjunc in conjuncs[1:]:
            op = algebra.Select(conjunc, op)
            op.has_been_pushed = False
        return op

    def __str__(self):
        return "Select => Select, Select"


class PushSelects(Rule):
    """Push selections."""

    @staticmethod
    def descend_tree(op, cond):
        """Recursively push a selection condition down a tree of operators.

        :param op: The root of an operator tree
        :type op: raco.algebra.Operator
        :type cond: The selection condition
        :type cond: raco.expression.expression

        :return: A (possibly modified) operator.
        """

        if isinstance(op, algebra.Select):
            # Keep pushing; selects are commutative
            op.input = PushSelects.descend_tree(op.input, cond)
            return op
        elif isinstance(op, algebra.CompositeBinaryOperator):
            # Joins and cross-products; consider conversion to an equijoin
            left_len = len(op.left.scheme())
            accessed = accessed_columns(cond)
            in_left = [col < left_len for col in accessed]
            if all(in_left):
                # Push the select into the left sub-tree.
                op.left = PushSelects.descend_tree(op.left, cond)
                return op
            elif not any(in_left):
                # Push into right subtree; rebase column indexes
                expression.rebase_expr(cond, left_len)
                op.right = PushSelects.descend_tree(op.right, cond)
                return op
            else:
                # Selection includes both children; attempt to create an
                # equijoin condition
                cols = is_column_equality_comparison(cond)
                if cols:
                    return op.add_equijoin_condition(cols[0], cols[1])
        elif isinstance(op, algebra.Apply):
            # Convert accessed to a list from a set to ensure consistent order
            accessed = list(accessed_columns(cond))
            accessed_emits = [op.emitters[i][1] for i in accessed]
            if all(isinstance(e, expression.AttributeRef)
                   for e in accessed_emits):
                unnamed_emits = [expression.toUnnamed(e, op.input.scheme())
                                 for e in accessed_emits]
                # This condition only touches columns that are copied verbatim
                # from the child, so we can push it.
                index_map = {a: e.position
                             for (a, e) in zip(accessed, unnamed_emits)}
                expression.reindex_expr(cond, index_map)
                op.input = PushSelects.descend_tree(op.input, cond)
                return op
        elif isinstance(op, algebra.GroupBy):
            # Convert accessed to a list from a set to ensure consistent order
            accessed = list(accessed_columns(cond))
            if all((a < len(op.grouping_list)) for a in accessed):
                accessed_grps = [op.grouping_list[a] for a in accessed]
                # This condition only touches columns that are copied verbatim
                # from the child (grouping keys), so we can push it.
                assert all(isinstance(e, expression.AttributeRef)
                           for e in op.grouping_list)
                unnamed_grps = [expression.toUnnamed(e, op.input.scheme())
                                for e in accessed_grps]
                index_map = {a: e.position
                             for (a, e) in zip(accessed, unnamed_grps)}
                expression.reindex_expr(cond, index_map)
                op.input = PushSelects.descend_tree(op.input, cond)
                return op

        # Can't push any more: instantiate the selection
        new_op = algebra.Select(cond, op)
        new_op.has_been_pushed = True
        return new_op

    def fire(self, op):
        if not isinstance(op, algebra.Select):
            return op
        if op.has_been_pushed:
            return op

        new_op = PushSelects.descend_tree(op.input, op.condition)

        # The new root may also be a select, so fire the rule recursively
        return self.fire(new_op)

    def __str__(self):
        return ("Select, Cross/Join => Join;"
                + " Select, Apply => Apply, Select;"
                + " Select, GroupBy => GroupBy, Select")


class MergeSelects(Rule):
    """Merge consecutive Selects into a single conjunctive selection."""

    def fire(self, op):
        if not isinstance(op, algebra.Select):
            return op

        while isinstance(op.input, algebra.Select):
            conjunc = expression.AND(op.condition, op.input.condition)
            op = algebra.Select(conjunc, op.input.input)

        return op

    def __str__(self):
        return "Select, Select => Select"


class PushApply(Rule):
    """Many Applies in MyriaL are added to select fewer columns from the
    input. In some  of these cases, we can do less work in the children by
    preventing them from producing columns we will then immediately drop.

    Currently, this rule:
      - merges consecutive Apply operations into one Apply, possibly dropping
        some of the produced columns along the way.
      - makes ProjectingJoin only produce columns that are later read.
        TODO: drop the Apply if the column-selection pushed into the
        ProjectingJoin is everything the Apply was doing. See note below.
    """

    def fire(self, op):
        if not isinstance(op, algebra.Apply):
            return op

        child = op.input

        if isinstance(child, algebra.Apply):
            in_scheme = child.scheme()
            child_in_scheme = child.input.scheme()
            names, emits = zip(*op.emitters)
            emits = [to_unnamed_recursive(e, in_scheme)
                     for e in emits]
            child_emits = [to_unnamed_recursive(e[1], child_in_scheme)
                           for e in child.emitters]

            def convert(n):
                if isinstance(n, expression.UnnamedAttributeRef):
                    n = child_emits[n.position]
                else:
                    n.apply(convert)
                return n

            emits = [convert(copy.deepcopy(e)) for e in emits]

            new_apply = algebra.Apply(emitters=zip(names, emits),
                                      input=child.input)
            return self.fire(new_apply)

        elif isinstance(child, algebra.ProjectingJoin):
            in_scheme = child.scheme()
            names, emits = zip(*op.emitters)
            emits = [to_unnamed_recursive(e, in_scheme)
                     for e in emits]
            accessed = sorted(set(itertools.chain(*(accessed_columns(e)
                                                    for e in emits))))
            index_map = {a: i for (i, a) in enumerate(accessed)}
            child.output_columns = [child.output_columns[i] for i in accessed]
            for e in emits:
                expression.reindex_expr(e, index_map)
            # TODO(dhalperi) we may not need the Apply if all it did was rename
            # and/or select certain columns. Figure out these cases and omit
            # the Apply
            return algebra.Apply(emitters=zip(names, emits),
                                 input=child)

        return op

    def __str__(self):
        return 'Push Apply into Apply, ProjectingJoin'


class RemoveUnusedColumns(Rule):
    """For operators that construct new tuples (e.g., GroupBy or Join), we are
    guaranteed that any columns from an input tuple that are ignored (neither
    used internally nor to produce the output columns) cannot be used higher
    in the query tree. For these cases, this rule will prepend an Apply that
    keeps only the referenced columns. The goal is that after this rule,
    a subsequent invocation of PushApply will be able to push that
    column-selection operation further down the tree."""

    def fire(self, op):
        if isinstance(op, algebra.GroupBy):
            child = op.input
            child_scheme = child.scheme()
            grp_list = [to_unnamed_recursive(g, child_scheme)
                        for g in op.grouping_list]
            agg_list = [to_unnamed_recursive(a, child_scheme)
                        for a in op.aggregate_list]
            agg = [accessed_columns(a) for a in agg_list]
            pos = [g.position for g in grp_list]
            accessed = sorted(set(itertools.chain(*(agg + [pos]))))
            if not accessed:
                # Bug #207: COUNTALL() does not access any columns. So if the
                # query is just a COUNT(*), we would generate an empty Apply.
                # If this happens, just keep the first column of the input.
                accessed = [0]
            if len(accessed) != len(child_scheme):
                emitters = [(None, UnnamedAttributeRef(i)) for i in accessed]
                new_apply = algebra.Apply(emitters, child)
                index_map = {a: i for (i, a) in enumerate(accessed)}
                for agg_expr in itertools.chain(grp_list, agg_list):
                    expression.reindex_expr(agg_expr, index_map)
                op.grouping_list = grp_list
                op.aggregate_list = agg_list
                op.input = new_apply
                return op
        elif isinstance(op, algebra.ProjectingJoin):
            l_scheme = op.left.scheme()
            r_scheme = op.right.scheme()
            in_scheme = l_scheme + r_scheme
            condition = to_unnamed_recursive(op.condition, in_scheme)
            column_list = [to_unnamed_recursive(c, in_scheme)
                           for c in op.output_columns]

            accessed = (accessed_columns(condition)
                        | set(c.position for c in op.output_columns))
            if len(accessed) == len(in_scheme):
                return op

            accessed = sorted(accessed)
            left = [a for a in accessed if a < len(l_scheme)]
            if len(left) < len(l_scheme):
                emits = [(None, UnnamedAttributeRef(a)) for a in left]
                apply = algebra.Apply(emits, op.left)
                op.left = apply
            right = [a - len(l_scheme) for a in accessed
                     if a >= len(l_scheme)]
            if len(right) < len(r_scheme):
                emits = [(None, UnnamedAttributeRef(a)) for a in right]
                apply = algebra.Apply(emits, op.right)
                op.right = apply
            index_map = {a: i for (i, a) in enumerate(accessed)}
            expression.reindex_expr(condition, index_map)
            [expression.reindex_expr(c, index_map) for c in column_list]
            op.condition = condition
            op.output_columns = column_list
            return op

        return op

    def __str__(self):
        return 'Remove unused columns'