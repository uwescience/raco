from raco import algebra
from pipelines import Pipelined
from raco.utility import emit

import logging
LOG = logging.getLogger(__name__)

"""
Apply rules to an expression
If successful, output will
only involve operators in the
target algebra
"""


def optimize_by_rules(expr, rules):
    for rule in rules:
        def recursiverule(expr):
            newexpr = rule(expr)
            LOG.debug("apply rule %s\n--- %s => %s", rule, expr, newexpr)
            newexpr.apply(recursiverule)
            return newexpr
        expr = recursiverule(expr)
    return expr


def optimize(expr, target, source, eliminate_common_subexpressions=False):
    """Fire the rule-based optimizer on an expression.  Fire all rules in the
    source algebra (logical) and the target algebra (physical)"""
    assert isinstance(expr, algebra.Operator)

    def opt(expr):
        so = optimize_by_rules(expr, source.opt_rules())
        newexpr = optimize_by_rules(so, target.opt_rules())
        if eliminate_common_subexpressions:
            newexpr = common_subexpression_elimination(newexpr)
        return newexpr
    return opt(expr)


def compile(expr):
    """Compile physical plan to linearized form for execution"""
    # TODO: Fix this
    algebra.reset()
    exprcode = []

    # TODO, actually use Parallel[Store...]]? Right now assumes it
    assert isinstance(expr, algebra.Parallel), "expected Parallel toplevel only"  # noqa
    assert len(expr.children()) == 1, "expected single expression only"
    store_expr = expr.children()[0]
    assert isinstance(store_expr, algebra.Store)
    assert len(store_expr.children()) == 1, "expected single expression only"  # noqa

    lang = store_expr.language

    if isinstance(store_expr, Pipelined):
        body = lang.body(store_expr.compilePipeline())
    else:
        body = lang.body(expr)

    exprcode.append(emit(body))
    return emit(*exprcode)


def search(expr, tofind):
    """yield a sequence of subexpressions equal to tofind"""
    def match(node):
        if node == tofind:
            yield node
    for x in expr.preorder(match):
        yield x


def common_subexpression_elimination(expr):
    """remove redundant subexpressions"""
    def id(expr):
        yield expr
    eqclasses = []
    allfound = []
    for x in expr.preorder(id):
        if x not in allfound:
            found = [x for x in search(expr, x)]
            eqclasses.append((x, found))
            allfound += found

    def replace(expr):
        for witness, ec in eqclasses:
            if expr in ec:
                expr.apply(replace)
                # record the fact that we eliminated the redundant branches
                if witness != expr:
                    # witness.trace("replaces", expr)
                    for k, v in expr.gettrace():
                        witness.trace(k, v)
                return witness

    return expr.apply(replace)
