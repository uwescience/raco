from raco import algebra
import raco.backends as language
from .pipelines import Pipelined
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


def print_pretty_plan(plan, indent=0):
    if isinstance(plan, algebra.DoWhile):
        children = plan.children()
        body = children[:-1]
        term = children[-1]

        spc = ' ' * indent
        print '%sDO' % spc
        for op in body:
            print_pretty_plan(op, indent + 4)
        print '%sWHILE' % spc
        print_pretty_plan(term, indent + 4)
    elif isinstance(plan, algebra.Sequence):
        print '%s%s' % (' ' * indent, plan.shortStr())
        for child in plan.children():
            print_pretty_plan(child, indent + 4)
    else:
        print '%s%s' % (' ' * indent, plan)

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
        expr = rule(expr)

    return expr


def optimize(expr, target, **kwargs):
    """Fire the rule-based optimizer on an expression.  Fire all rules in the
    target algebra."""
    assert isinstance(expr, algebra.Operator)
    assert isinstance(target, language.Algebra), type(target)

    return optimize_by_rules(expr, target.opt_rules(**kwargs))


def compile(expr, **kwargs):
    """Compile physical plan to linearized form for execution"""
    # TODO: Fix this --> what?
    algebra.reset()
    exprcode = []

    # fixup for legacy usage of compile
    if not hasattr(expr, 'language'):
        assert isinstance(expr, algebra.Sequence) \
            or isinstance(expr, algebra.Parallel)
        assert len(expr.args) == 1
        expr = expr.args[0]

    lang = expr.language()

    if isinstance(expr, Pipelined):
        body = lang.body(expr.compilePipeline(**kwargs))
    else:
        body = lang.body(expr)

    exprcode.append(emit(body))
    return emit(*exprcode)
