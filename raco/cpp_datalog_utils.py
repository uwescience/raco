from raco import RACompiler
from raco.compile import compile

from raco.backends.radish import GrappaSymmetricHashJoin
from raco.backends.radish import GrappaShuffleHashJoin

import raco.viz as viz

import logging
LOG = logging.getLogger(__name__)


def comment(s):
    return "/*\n%s\n*/\n" % str(s)


def hack_plan(alg, plan):
    # plan hacking
    newRule = None
    if plan == "sym":
        alg.set_join_type(GrappaSymmetricHashJoin)
    elif plan == "shuf":
        alg.set_join_type(GrappaShuffleHashJoin)


def emitCode(query, name, algType, plan=None, emit_print=None, dir='.'):
    if emit_print is not None:
        alg = algType(emit_print)
    else:
        alg = algType()

    hack_plan(alg, plan)

    LOG.info("compiling %s: %s", name, query)

    # Create a compiler object
    dlog = RACompiler()

    # parse the query
    dlog.fromDatalog(query)
    # print dlog.parsed
    LOG.info("logical: %s", dlog.logicalplan)

    print dlog.logicalplan
    logical_dot = viz.operator_to_dot(dlog.logicalplan)
    with open("%s.logical.dot" % (name), 'w') as dwf:
        dwf.write(logical_dot)

    dlog.optimize(target=alg)

    LOG.info("physical: %s", dlog.physicalplan)

    print dlog.physicalplan
    physical_dot = viz.operator_to_dot(dlog.physicalplan)
    with open("%s.physical.dot" % (name), 'w') as dwf:
        dwf.write(physical_dot)

    # generate code in the target language
    code = ""
    code += comment("Query " + query)
    code += compile(dlog.physicalplan)

    fname = '{dir}/{name}.cpp'.format(dir=dir, name=name)
    with open(fname, 'w') as f:
        f.write(code)

    # returns name of code file
    return fname
