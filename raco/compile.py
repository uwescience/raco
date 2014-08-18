from raco import algebra
import raco.language as language
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


def optimize(expr, target, source, **kwargs):
    """Fire the rule-based optimizer on an expression.  Fire all rules in the
    source algebra (logical) and the target algebra (physical)"""
    assert isinstance(expr, algebra.Operator)
    assert isinstance(target, language.Algebra), type(target)
    assert isinstance(source, language.Algebra), type(source)

    so = optimize_by_rules(expr, source.opt_rules())
    return optimize_by_rules(so, target.opt_rules())


def compile(expr):
    """Compile physical plan to linearized form for execution"""
    # TODO: Fix this
    algebra.reset()
    exprcode = []

    # TODO, actually use Parallel[Store...]]? Right now assumes it
    if isinstance(expr, (algebra.Sequence, algebra.Parallel)):
        assert len(expr.children()) == 1, "expected single expression only"
        store_expr = expr.children()[0]
    else:
        store_expr = expr

    assert isinstance(store_expr, algebra.Store)
    assert len(store_expr.children()) == 1, "expected single expression only"

    lang = store_expr.language

    if isinstance(store_expr, Pipelined):
        body = lang.body(store_expr.compilePipeline())
    else:
        body = lang.body(expr)

    exprcode.append(emit(body))
    return emit(*exprcode)