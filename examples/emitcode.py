from raco import RACompiler
from raco.algebra import LogicalAlgebra
import raco.algebra as algebra
from raco.compile import compile
from raco.grappalang import GrappaShuffleHashJoin, GrappaSymmetricHashJoin, GrappaHashJoin
import raco.rules as rules
import generateDot

import logging
LOG = logging.getLogger(__name__)

def comment(s):
  return "/*\n%s\n*/\n" % str(s)

def hack_plan(alg, plan):
    # plan hacking
    newRule = None
    if plan == "sym":
        newRule = rules.OneToOne(algebra.Join, GrappaSymmetricHashJoin)
    elif plan == "shuf":
        newRule = rules.OneToOne(algebra.Join, GrappaShuffleHashJoin)
    
    if newRule:
        for i in range(0, len(alg.rules)):
            r = alg.rules[i]
            if isinstance(r, rules.OneToOne) and r.opto == GrappaHashJoin:
                alg.rules[i] = newRule

def emitCode(query, name, alg, plan=""):
    hack_plan(alg, plan)

    LOG.info("compiling %s: %s", name, query)

    # Create a compiler object
    dlog = RACompiler()

    # parse the query
    dlog.fromDatalog(query)
    #print dlog.parsed
    LOG.info("logical: %s",dlog.logicalplan)

    generateDot.generateDot(dlog.logicalplan, "%s.logical.dot"%(name))

    dlog.optimize(target=alg, eliminate_common_subexpressions=False)

    LOG.info("physical: %s",dlog.physicalplan[0][1])
    
    generateDot.generateDot(dlog.physicalplan, "%s.physical.dot"%(name))

    # generate code in the target language
    code = ""
    code += comment("Query " + query)
    code += compile(dlog.physicalplan)
    
    fname = name+'.cpp'
    with open(fname, 'w') as f:
        f.write(code)

    # returns name of code file
    return fname

