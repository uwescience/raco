from raco import algebra
import raco.language as language
from pipelines import Pipelined
from raco.utility import emit
import raco.viz as viz
import os
from raco.utility import colored

import logging
LOG = logging.getLogger(__name__)


"""
Apply rules to an expression
If successful, output will
only involve operators in the
target algebra
"""


class PlanWriter():

    def __init__(self, template="wip-%02d.physical.dot", limit=20):
        self.ind = 0
        self.template = template
        self.limit = limit
        self.enabled = os.environ.get('RACO_OPTIMIZER_GRAPHS') in \
            ['true', 'True', 't', 'T', '1', 'yes', 'y']

    def write_if_enabled(self, plan, title):
        if self.enabled:
            with open(self.template % self.ind, 'w') as dwf:
                dwf.write(viz.operator_to_dot(plan, title=title))

        self.ind += 1


def optimize_by_rules(expr, rules):
    writer = PlanWriter()
    writer.write_if_enabled(expr, "before rules")

    for rule in rules:
        def recursiverule(e):
            newe = rule(e)
            writer.write_if_enabled(newe, str(rule))

            LOG.debug("apply rule %s\n" +
                      colored("  -", "red") + " %s" + "\n" +
                      colored("  +", "green") + " %s", rule, e, newe)
            newe.apply(recursiverule)

            return newe
        expr = recursiverule(expr)

    return expr


def optimize(expr, target, **kwargs):
    """Fire the rule-based optimizer on an expression.  Fire all rules in the
    target algebra."""
    assert isinstance(expr, algebra.Operator)
    assert isinstance(target, language.Algebra), type(target)

    return optimize_by_rules(expr, target.opt_rules(**kwargs))


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

    lang = store_expr.language()

    if isinstance(store_expr, Pipelined):
        body = lang.body(store_expr.compilePipeline())
    else:
        body = lang.body(expr)

    exprcode.append(emit(body))
    return emit(*exprcode)