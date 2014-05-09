from raco import RACompiler
from raco.algebra import LogicalAlgebra
from raco.compile import compile
import generateDot

import logging
LOG = logging.getLogger(__name__)

def comment(s):
  return "/*\n%s\n*/\n" % str(s)

def emitCode(query, name, algebra):
    LOG.info("compiling %s: %s", name, query)

    # Create a compiler object
    dlog = RACompiler()

    # parse the query
    dlog.fromDatalog(query)
    #print dlog.parsed
    LOG.info("logical: %s",dlog.logicalplan)

    generateDot.generateDot(dlog.logicalplan, "%s.logical.dot"%(name))

    dlog.optimize(target=algebra, eliminate_common_subexpressions=False)

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
