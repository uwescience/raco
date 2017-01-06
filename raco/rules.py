import re

from raco import algebra, expression
from raco.representation import RepresentationProperties
from .expression import (accessed_columns, UnnamedAttributeRef,
                         rebase_local_aggregate_output, rebase_finalizer,
                         to_unnamed_recursive, StateVar, RANDOM)

from abc import ABCMeta, abstractmethod
import itertools


class Rule(object):

    """Argument is an expression tree

    Returns a possibly modified expression tree"""

    __metaclass__ = ABCMeta

    _flag_pattern = re.compile(r'no_([A-Za-z_]+)')  # e.g., no_MergeSelects

    def __init__(self):
        self._disabled = False

    def __call__(self, expr):
        if self._disabled:
            return expr
        else:
            return self.fire(expr)

    @classmethod
    def apply_disable_flags(cls, rule_list, *args):
        disabled_rules = set()
        # Automatically create a flag to disable any rule by name
        # e.g., to disable MergeSelects, pass the arg "no_MergeSelects"
        for a in args:
            mat = re.match(cls._flag_pattern, a)
            if mat:
                disabled_rules.add(mat.group(1))

        for r in rule_list:
            r._disabled = r.__class__.__name__ in disabled_rules

    @abstractmethod
    def fire(self, expr):
        """Apply this rule to the supplied expression tree"""


class AbstractInterpretedValue:

    def __init__(self):
        self._values = set()

    def appendValue(self, v):
        self._values.add(v)

    def setValues(self, values):
        self._values = set(values)

    def getValues(self):
        return self._values


class NumTuplesPropagation(Rule):

    def fire(self, expr):
        # TODO I really just want this to fire once on the top node...
        if isinstance(expr, algebra.Sequence):
            self._num_tuples_analysis(expr)
        return expr

    def _num_tuples_analysis(self, tree):
        """Do an abstract interpretation of the tree to propagate StoreTemp
        num_tuples to corresponding ScanTemps. This analysis mutates
        the tree by tagging GrappaMemoryScan"""

        # Right now, the abstract interpretation is not too interesting,
        # as we ignore loops.
        abstract_values = {}

        def f(t):
            if isinstance(t, algebra.StoreTemp):
                if t.name not in abstract_values:
                    abstract_values[t.name] = AbstractInterpretedValue()

                abstract_values[t.name].appendValue(t.num_tuples())
            elif isinstance(t, algebra.ScanTemp):
                if t.name:
                    assert t.name in abstract_values, "Saw a ScanTemp before " \
                                                      "its StoreTemp"
                    possible_values = \
                        abstract_values[t.name].getValues()
                    if len(possible_values) == 1:
                        for v in possible_values:
                            t.analyzed_num_tuples = v

            yield None

        # run it
        [_ for _ in tree.postorder(f)]


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
        super(OneToOne, self).__init__()

    def fire(self, expr):
        if isinstance(expr, self.opfrom):
            newop = self.opto()
            newop.copy(expr)
            return newop
        return expr

    def __str__(self):
        return "%s => %s" % (self.opfrom.__name__, self.opto.__name__)


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


def is_simple_agg_expr(agg):
    """A simple aggregate expression is an aggregate whose input is an
    AttributeRef."""
    if isinstance(agg, expression.COUNTALL):
        return True
    elif isinstance(agg, expression.UdaAggregateExpression):
        return True
    elif (isinstance(agg, expression.UnaryOperator) and
          isinstance(agg, expression.BuiltinAggregateExpression) and
          isinstance(agg.input, expression.AttributeRef)):
        return True
    return False


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

        # A simple grouping expression is an AttributeRef
        def is_simple_grp_expr(grp):
            return isinstance(grp, expression.AttributeRef)

        complex_grp_exprs = [(i, grp)
                             for (i, grp) in enumerate(expr.grouping_list)
                             if not is_simple_grp_expr(grp)]

        complex_agg_exprs = [agg for agg in expr.aggregate_list
                             if not is_simple_agg_expr(agg)]

        # There are no complicated expressions, we're okay with the existing
        # GroupBy.
        if not complex_grp_exprs and not complex_agg_exprs:
            return expr

        # Construct the Apply we're going to stick before the GroupBy

        child_scheme = expr.input.scheme()

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


class DedupGroupBy(Rule):

    """When a GroupBy computes redundant fields, replace this duplicate
    computation by a single computation plus a duplicating Apply."""

    def fire(self, expr):
        if not isinstance(expr, algebra.GroupBy):
            return expr

        aggs = expr.get_unnamed_aggregate_list()

        # Maps aggregate index j to index i < j that j duplicates
        dups = {}
        # Maps non-dup aggregate index i to index i' <= i it will have
        # in the output aggregate list.
        orig = {}
        for i, a in enumerate(aggs):
            if i in dups:
                continue
            orig[i] = len(orig)
            dups.update({(j + i + 1): i
                         for j, b in enumerate(aggs[(i + 1):])
                         if a == b})

        if len(dups) == 0:
            # All the aggregate expressions are unique, we're good
            return expr

        #################################
        # Construct a new Apply that drops all duplicates and replaces
        # them with repeated UnnamedAttributeRefs
        #################################

        # First keep the grouping list intact
        num_grps = len(expr.grouping_list)
        mappings = [(None, UnnamedAttributeRef(i))
                    for i in range(num_grps)]

        # Construct the references to the grouping list
        for i in range(len(aggs)):
            if i in orig:
                m = orig[i] + num_grps
            else:
                m = orig[dups[i]] + num_grps
            mappings.append((None, UnnamedAttributeRef(m)))

        # Drop any duplicates from the agg list
        expr.aggregate_list = [aggs[i] for i in sorted(orig)]

        return algebra.Apply(emitters=mappings, input=expr)


class DistinctToGroupBy(Rule):

    """Turns a distinct into an empty GroupBy"""

    def fire(self, expr):
        if isinstance(expr, algebra.Distinct):
            in_scheme = expr.scheme()
            grps = [UnnamedAttributeRef(i) for i in range(len(in_scheme))]
            return algebra.GroupBy(input=expr.input, grouping_list=grps)

        return expr

    def __str__(self):
        return "Distinct => GroupBy(no groupings)"


class EmptyGroupByToDistinct(Rule):

    """Turns a GroupBy with no aggregates into a Distinct"""

    def fire(self, expr):
        if isinstance(expr, algebra.GroupBy) and len(expr.aggregate_list) == 0:
            # We can turn an empty GroupBy into a Distinct. However,
            # we must ensure that the GroupBy does not do any column
            # re-ordering.
            group_cols = expr.get_unnamed_grouping_list()
            if all(e.position == i for i, e in enumerate(group_cols)):
                # No reordering is done
                return algebra.Distinct(input=expr.input)

            # Some reordering is done, so shim in the Apply to mimic it.
            reorder_cols = algebra.Apply(
                emitters=[(None, e) for e in group_cols], input=expr.input)
            return algebra.Distinct(input=reorder_cols)

        return expr

    def __str__(self):
        return "GroupBy(no groupings) => Distinct"


class CountToCountall(Rule):

    """Since Raco does not support NULLs at the moment, it is safe to always
    map COUNT to COUNTALL."""
    # TODO fix when we have NULL support.

    def fire(self, expr):
        if not isinstance(expr, algebra.GroupBy):
            return expr

        assert all(is_simple_agg_expr(agg) for agg in expr.aggregate_list)

        counts = [i for (i, agg) in enumerate(expr.aggregate_list)
                  if isinstance(agg, expression.COUNT)]
        if not counts:
            return expr

        for i in counts:
            expr.aggregate_list[i] = expression.COUNTALL()

        return expr

    def __str__(self):
        return "Count(x) => Countall()"


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

        conjuncs = expression.extract_conjuncs(op.get_unnamed_condition())
        assert conjuncs  # Must be at least 1

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
    def is_column_equality_comparison(cond):
        """Return a tuple of column indexes if the condition is an equality
        test.
        """

        if (isinstance(cond, expression.EQ) and
                isinstance(cond.left, UnnamedAttributeRef) and
                isinstance(cond.right, UnnamedAttributeRef)):
            return cond.left.position, cond.right.position
        else:
            return None

    @staticmethod
    def descend_tree(op, cond):
        """Recursively push a selection condition down a tree of operators.

        :param op: The root of an operator tree
        :type op: raco.algebra.Operator
        :param cond: The selection condition
        :type cond: raco.expression.expression

        :return: A (possibly modified) operator.
        """

        if isinstance(op, algebra.Select):
            # Keep pushing; selects are commutative
            op.input = PushSelects.descend_tree(op.input, cond)
            return op
        elif isinstance(op, algebra.CompositeBinaryOperator):
            # Joins and cross-products; consider conversion to an equijoin
            # Expressions containing random do not commute across joins
            has_random = any(isinstance(e, RANDOM) for e in cond.walk())
            if not has_random:
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
                    cols = PushSelects.is_column_equality_comparison(cond)
                    if cols:
                        return op.add_equijoin_condition(cols[0], cols[1])
        elif isinstance(op, algebra.Apply):
            # Convert accessed to a list from a set to ensure consistent order
            accessed = list(accessed_columns(cond))
            accessed_emits = [op.emitters[i][1] for i in accessed]
            if all(isinstance(e, expression.AttributeRef)
                   for e in accessed_emits):
                unnamed_emits = expression.ensure_unnamed(
                    accessed_emits, op.input)
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
                unnamed_grps = expression.ensure_unnamed(accessed_grps,
                                                         op.input)
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
        if hasattr(op, "has_been_pushed"):
            if op.has_been_pushed:
                return op
        else:
            op.has_been_pushed = False

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
    """

    def fire(self, op):
        if not isinstance(op, algebra.Apply):
            return op

        child = op.input

        if isinstance(child, algebra.Apply):

            emits = op.get_unnamed_emit_exprs()
            child_emits = child.get_unnamed_emit_exprs()

            def convert(n):
                if isinstance(n, expression.UnnamedAttributeRef):
                    return child_emits[n.position]
                n.apply(convert)
                return n
            emits = [convert(e) for e in emits]

            new_apply = algebra.Apply(emitters=zip(op.get_names(), emits),
                                      input=child.input)
            return self.fire(new_apply)

        elif isinstance(child, algebra.ProjectingJoin):
            emits = op.get_unnamed_emit_exprs()

            # If this apply is only AttributeRefs and the columns already
            # have the correct names, we can push it into the ProjectingJoin
            if (all(isinstance(e, expression.AttributeRef) for e in emits) and
                    len(set(emits)) == len(emits)):
                new_cols = [child.output_columns[e.position] for e in emits]
                # We need to ensure that left columns come before right cols
                left_sch = child.left.scheme()
                right_sch = child.right.scheme()
                combined = left_sch + right_sch
                left_len = len(left_sch)
                new_cols = [expression.to_unnamed_recursive(e, combined)
                            for e in new_cols]
                side = [e.position >= left_len for e in new_cols]
                if sorted(side) == side:
                    # Left columns do come before right cols
                    new_pj = algebra.ProjectingJoin(
                        condition=child.condition, left=child.left,
                        right=child.right, output_columns=new_cols)
                    if new_pj.scheme() == op.scheme():
                        return new_pj

            accessed = sorted(set(itertools.chain(*(accessed_columns(e)
                                                    for e in emits))))
            index_map = {a: i for (i, a) in enumerate(accessed)}
            child.output_columns = [child.output_columns[i] for i in accessed]
            for e in emits:
                expression.reindex_expr(e, index_map)

            return algebra.Apply(emitters=zip(op.get_names(), emits),
                                 input=child)

        elif isinstance(child, algebra.GroupBy):
            emits = op.get_unnamed_emit_exprs()
            assert all(is_simple_agg_expr(agg) for agg in child.aggregate_list)

            accessed = sorted(set(itertools.chain(*(accessed_columns(e)
                                                    for e in emits))))
            num_grps = len(child.grouping_list)
            accessed_aggs = [i for i in accessed if i >= num_grps]
            if len(accessed_aggs) == len(child.aggregate_list):
                return op

            unused_map = {i: j + num_grps for j, i in enumerate(accessed_aggs)}

            # copy the groupby operator so we can modify it
            newgb = child.__class__()
            newgb.copy(child)

            # remove aggregates that are projected out
            newgb.aggregate_list = [newgb.aggregate_list[i - num_grps]
                                    for i in accessed_aggs]
            for e in emits:
                expression.reindex_expr(e, unused_map)

            return algebra.Apply(emitters=zip(op.get_names(), emits),
                                 input=newgb)

        return op

    def __str__(self):
        return 'Push Apply into Apply, ProjectingJoin'


class ProjectToDistinctColumnSelect(Rule):

    def fire(self, expr):
        # If not a Project, who cares?
        if not isinstance(expr, algebra.Project):
            return expr

        mappings = [(None, x) for x in expr.columnlist]
        col_select = algebra.Apply(mappings, expr.input)
        return algebra.Distinct(input=col_select)

    def __str__(self):
        return 'Project => Distinct, Column select'


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
            grp_list = op.get_unnamed_grouping_list()
            agg_list = op.get_unnamed_aggregate_list()

            up_names = [name for name, ex in op.updaters]
            up_list = op.get_unnamed_update_exprs()

            agg = [accessed_columns(a) for a in agg_list]
            up = [accessed_columns(a) for a in up_list]
            pos = [{g.position} for g in grp_list]

            accessed = sorted(set(itertools.chain(*(up + agg + pos))))
            if not accessed:
                # Bug #207: COUNTALL() does not access any columns. So if the
                # query is just a COUNT(*), we would generate an empty Apply.
                # If this happens, just keep the first column of the input.
                accessed = [0]
            if len(accessed) != len(child_scheme):
                emitters = [(None, UnnamedAttributeRef(i)) for i in accessed]
                new_apply = algebra.Apply(emitters, child)
                index_map = {a: i for (i, a) in enumerate(accessed)}
                for agg_expr in itertools.chain(grp_list, agg_list, up_list):
                    expression.reindex_expr(agg_expr, index_map)
                op.grouping_list = grp_list
                op.aggregate_list = agg_list
                op.updaters = [(name, ex) for name, ex in
                               zip(up_names, up_list)]
                op.input = new_apply
                return op
        elif isinstance(op, algebra.ProjectingJoin):
            l_scheme = op.left.scheme()
            r_scheme = op.right.scheme()
            in_scheme = l_scheme + r_scheme
            condition = to_unnamed_recursive(op.condition, in_scheme)
            column_list = [to_unnamed_recursive(c, in_scheme)
                           for c in op.output_columns]

            accessed = (accessed_columns(condition) |
                        set(c.position for c in op.output_columns))
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


class ProjectingJoinToProjectOfJoin(Rule):

    """Turn ProjectingJoin to Project of a Join.
    This is useful to take advantage of the column selection
    optimizations and then remove ProjectingJoin for
    backends that don't have one"""

    def fire(self, expr):
        if isinstance(expr, algebra.ProjectingJoin):
            return algebra.Apply([(None, x) for x in expr.output_columns],
                                 algebra.Join(expr.condition,
                                              expr.left,
                                              expr.right))

        return expr

    def __str__(self):
        return 'ProjectingJoin[$1] => Project[$1](Join)'


class RemoveNoOpApply(Rule):

    """Remove Apply operators that have no effect."""

    def fire(self, op):
        if not isinstance(op, algebra.Apply):
            return op

        # At least one emit expression is not just copying a column
        if not all(isinstance(e[1], expression.AttributeRef)
                   for e in op.emitters):
            return op

        child = op.input
        child_scheme = child.scheme()

        # Schemes are different, this Apply does something
        if child_scheme != op.scheme():
            return op

        emitters = [expression.toUnnamed(e[1], child_scheme)
                    for e in op.emitters]
        # Schemes are the same (including names), and this Apply keeps all
        # columns in the same order. This Apply does nothing.
        if all(e.position == i for (i, e) in enumerate(emitters)):
            return child

        return op

    def __str__(self):
        return 'Remove no-op apply'


class SwapJoinSides(Rule):
    # swaps the inputs to a join

    def fire(self, expr):
        # don't allow swap-created join to be swapped
        if (isinstance(expr, algebra.Join) or
            isinstance(expr, algebra.CrossProduct)) \
                and not hasattr(expr, '__swapped__'):

            assert (
                isinstance(
                    expr,
                    algebra.Join)) or (
                isinstance(
                    expr,
                    algebra.CrossProduct))

            # An apply will undo the effect of the swap on the scheme,
            # so above operators won't be affected
            left_sch = expr.left.scheme()
            right_sch = expr.right.scheme()
            leftlen = len(left_sch)
            rightlen = len(right_sch)
            assert leftlen + rightlen == len(expr.scheme())
            emitters_left = [(left_sch.getName(i),
                              UnnamedAttributeRef(rightlen + i))
                             for i in range(leftlen)]
            emitters_right = [(right_sch.getName(i), UnnamedAttributeRef(i))
                              for i in range(rightlen)]
            emitters = emitters_left + emitters_right

            if isinstance(expr, algebra.Join):
                # reindex the expression
                index_map = dict([(oldpos, attr[1].position)
                                  for (oldpos, attr) in enumerate(emitters)])

                expression.reindex_expr(expr.condition, index_map)

                newjoin = algebra.Join(expr.condition, expr.right, expr.left)
            else:
                newjoin = algebra.CrossProduct(expr.right, expr.left)

            newjoin.__swapped__ = True

            return algebra.Apply(emitters=emitters, input=newjoin)
        else:
            return expr

    def __str__(self):
        return "Join(L,R) => Join(R,L)"


# logical groups of catalog transparent rules
# 1. this must be applied first
remove_trivial_sequences = [RemoveTrivialSequences()]

# 2. simple group by
simple_group_by = [SimpleGroupBy()]

# 3. push down selection
push_select = [
    SplitSelects(),
    PushSelects(),
    MergeSelects()
]

# 4. push projection
push_project = [
    ProjectToDistinctColumnSelect(),
    JoinToProjectingJoin()
]

# 5. push apply
push_apply = [
    # These really ought to be run until convergence.
    # For now, run twice and finish with PushApply.
    PushApply(),
    RemoveUnusedColumns(),
    PushApply(),
    RemoveUnusedColumns(),
    PushApply(),
    RemoveNoOpApply(),
]


class DecomposeGroupBy(Rule):

    """Convert a logical group by into a two-phase group by.

    The local half of the aggregate before the shuffle step, whereas the remote
    half runs after the shuffle step.

    TODO: omit this optimization if
          - the cardinality of the grouping keys is high.
    """

    def __init__(self, partition_groupby_class, only_fire_on_multi_key=None):
        self._gb_class = partition_groupby_class
        self._only_fire_on_multi_key = only_fire_on_multi_key
        super(DecomposeGroupBy, self).__init__()

    @staticmethod
    def do_transfer(op):
        """Introduce a network transfer before a groupby operation."""

        # Get an array of position references to columns in the child scheme
        child_scheme = op.input.scheme()
        group_fields = [expression.toUnnamed(ref, child_scheme)
                        for ref in op.grouping_list]
        if len(group_fields) == 0:
            # Need to Collect all tuples at once place
            op.input = algebra.Collect(op.input)
        else:
            # Need to Shuffle
            op.input = algebra.Shuffle(op.input, group_fields)

    @staticmethod
    def check_no_shuffle(op):
        """Check if no shuffle is needed"""

        # Get an array of position references to columns in the child scheme
        child_scheme = op.input.scheme()
        group_fields = [expression.toUnnamed(ref, child_scheme)
                        for ref in op.grouping_list]
        return (len(group_fields) > 0 and
                check_partition_equality(op.input, group_fields))

    def fire(self, op):
        # Punt if it's not a group by or we've already converted this into an
        # an instance of self.gb_class
        if op.__class__ != algebra.GroupBy:
            return op

        if self._only_fire_on_multi_key and len(op.grouping_list) == 0:
            out_op = self._only_fire_on_multi_key()
            out_op.copy(op)
            return out_op

        # Do not shuffle and do not decompose if the data is shuffled already
        if DecomposeGroupBy.check_no_shuffle(op):
            out_op = self._gb_class()
            out_op.copy(op)
            return out_op

        # Bail early if we have any non-decomposable aggregates
        if not all(x.is_decomposable() for x in op.aggregate_list):
            out_op = self._gb_class()
            out_op.copy(op)
            DecomposeGroupBy.do_transfer(out_op)
            return out_op

        num_grouping_terms = len(op.grouping_list)

        local_emitters = []
        local_statemods = []
        remote_emitters = []
        remote_statemods = []
        finalizer_exprs = []

        # The starting positions for the current local, remote aggregate
        local_output_pos = num_grouping_terms
        remote_output_pos = num_grouping_terms
        requires_finalizer = False

        for agg in op.aggregate_list:
            # Multiple emit arguments can be associated with a single
            # decomposition rule; coalesce them all together.
            state = agg.get_decomposable_state()
            assert state

            ################################
            # Extract the set of emitters and statemods required for the
            # local aggregate.
            ################################

            laggs = state.get_local_emitters()
            local_emitters.extend(laggs)
            local_statemods.extend(state.get_local_statemods())

            ################################
            # Extract the set of emitters and statemods required for the
            # remote aggregate.  Remote expressions must be rebased to
            # remove instances of LocalAggregateOutput
            ################################

            raggs = state.get_remote_emitters()
            raggs = [rebase_local_aggregate_output(x, local_output_pos)
                     for x in raggs]
            remote_emitters.extend(raggs)

            rsms = state.get_remote_statemods()
            for sm in rsms:
                update_expr = rebase_local_aggregate_output(
                    sm.update_expr, local_output_pos)
                remote_statemods.append(
                    StateVar(sm.name, sm.init_expr, update_expr))

            ################################
            # Extract any required finalizers.  These must be rebased to remove
            # instances of RemoteAggregateOutput
            ################################

            finalizer = state.get_finalizer()
            if finalizer is not None:
                requires_finalizer = True
                finalizer_exprs.append(
                    rebase_finalizer(finalizer, remote_output_pos))
            else:
                for i in range(len(raggs)):
                    finalizer_exprs.append(
                        UnnamedAttributeRef(remote_output_pos + i))

            local_output_pos += len(laggs)
            remote_output_pos += len(raggs)

        ################################
        # Glue together the local and remote aggregates:
        # Local => Shuffle => Remote => (optional) Finalizer.
        ################################

        local_gb = self._gb_class(op.grouping_list, local_emitters, op.input,
                                  local_statemods)

        grouping_fields = [UnnamedAttributeRef(i)
                           for i in range(num_grouping_terms)]

        remote_gb = self._gb_class(grouping_fields, remote_emitters, local_gb,
                                   remote_statemods)

        DecomposeGroupBy.do_transfer(remote_gb)

        if requires_finalizer:
            # Pass through grouping terms
            gmappings = [(None, UnnamedAttributeRef(i))
                         for i in range(num_grouping_terms)]
            fmappings = [(None, fx) for fx in finalizer_exprs]
            return algebra.Apply(gmappings + fmappings, remote_gb)
        return remote_gb


# 7. distributed groupby
# this need to be put after shuffle logic
def distributed_group_by(
        partition_groupby_class,
        countall_rule=True,
        **kwargs):
    r = [
        # DecomposeGroupBy may introduce a complex GroupBy,
        # so we must run SimpleGroupBy after it. TODO no one likes this.
        DecomposeGroupBy(partition_groupby_class, **kwargs),
        SimpleGroupBy()
    ]

    if countall_rule:
        r.append(CountToCountall())   # TODO revisit when we have NULL support.

    r += [
        DedupGroupBy(),
        EmptyGroupByToDistinct(),
    ]

    return r


def check_partition_equality(op, representation):
    """Check to see if the operator has the required hash partitioning.
    @param op operator
    @param representation list of columns hash partitioned by,
                        in the unnamed perspective
    @return true if the op has an equal hash partitioning to representation
    """

    return op.partitioning().hash_partitioned == frozenset(representation)


class DeDupBroadcastInputs(Rule):

    def fire(self, expr):
        def is_nonlocal_exchange_op(expr):
            return isinstance(expr, (
                algebra.Shuffle,
                algebra.HyperCubeShuffle,
                algebra.Collect,
                algebra.Broadcast,
            ))

        def debroadcast_scans(expr):
            assert expr.partitioning().broadcasted, \
                "Can only debroadcast broadcast inputs"
            for e in expr.walk():
                if isinstance(e, algebra.ZeroaryOperator):
                    e._debroadcast = True

        # We must evaluate broadcast partitioning of input to avoid asserts
        # that exchange operators are non-broadcast-partitioned.
        if (is_nonlocal_exchange_op(expr) and
                expr.input.partitioning().broadcasted):
            # eliminate redundant broadcasts
            if isinstance(expr, algebra.Broadcast):
                return expr.input
            # debroadcast all scans to avoid redundant inputs to shuffle
            debroadcast_scans(expr.input)
        return expr

    def __str__(self):
        return ("Broadcast(X = broadcast input) => X, "
                "X(broadcast input) => X(debroadcast input)")
