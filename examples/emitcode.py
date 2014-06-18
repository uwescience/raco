from raco import RACompiler
from raco.algebra import LogicalAlgebra
from raco.compile import compile
import raco.viz as viz

import logging
logging.basicConfig(level=logging.INFO)
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

    print dlog.logicalplan
    logical_dot = viz.operator_to_dot(dlog.logicalplan[0][1])
    with open("%s.logical.dot"%(name), 'w') as dwf:
        dwf.write(logical_dot)

    dlog.optimize(target=algebra, eliminate_common_subexpressions=False)

    LOG.info("physical: %s",dlog.physicalplan[0][1])
    
    print dlog.physicalplan
    physical_dot = viz.operator_to_dot(dlog.physicalplan[0][1])
    with open("%s.physical.dot"%(name), 'w') as dwf:
        dwf.write(physical_dot)

    # generate code in the target language
    code = ""
    code += comment("Query " + query)
    code += compile(dlog.physicalplan)
    
    fname = name+'.cpp'
    with open(fname, 'w') as f:
        f.write(code)

    # returns name of code file
    return fname
