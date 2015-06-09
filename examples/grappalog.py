from emitcode import emitCode
from raco.language import GrappaAlgebra
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
  alg = GrappaAlgebra
  prefix = "grappa"
  lst.append(prefix)
  if plan: lst.append(plan)
  if name: lst.append(name)
  emitCode(query, "_".join(lst), alg, plan)

