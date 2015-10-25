from raco.cpp_datalog_utils import emitCode
from raco.backends.cpp import CCAlgebra
import sys

import logging
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)
  
if __name__ == "__main__":
  query = sys.argv[1]
  print query
  name = sys.argv[2]
  print name

  plan = ""
  if len(sys.argv) > 3:
      plan = sys.argv[3]

  lst = []
  alg = CCAlgebra
  if plan: lst.append(plan)
  if name: lst.append(name)
  emitCode(query, "_".join(lst), alg, plan)

