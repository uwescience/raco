from raco.datalog.grammar import parse
from raco.language import CCAlgebra
from raco.algebra import LogicalAlgebra
from raco.compile import optimize
import scan_code_ver2 as sc
from generateDot import generateDot


import logging
LOG = logging.getLogger(__name__)
import sys

if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)

  if len(sys.argv) > 1:
      query = sys.argv[1]
      set_sem = bool(sys.argv[2])
      if (set_sem=='True'): set_sem = True
      else: set_sem=False
  else:
      set_sem = False
      #query = 'A(a1) :- R(a1,x),S(x,y),T(y,z),U(z,a1),100=z,y=50'
      #query = 'Triangle(x,y,z) :- R(x,y),S(y,z),T(z,x)'
      query = 'Triangle(x,y,z) :- edges(x,y),edges(y,z),edges(z,x),x<y,y<z'
      query = 'California(x,z) :- edges1(x,y1),edges1(y1,y2),edges2(y2,z)'
      #query = 'A(x,z) :- edges(x,y1),edges(y1,y2),edges(y2,z)'
      #query = 'we(a,c) :- edges(a,b),edges(b,c)'
      query = 'mutual(a,b) :- edges(a,b),edges(b,a)'
      #query = 'symmetric(a,b) :- edges1(a,b),edges2(b,a)'
      query = 'TwoPath(a,b,c) :- R(a,b),R(b,c)'

  import re
  p = re.compile('[^(]*')
  headname = p.match(query).group(0)

  print "query:", query, "\n"


  parsedprogram = parse(query)
  print "parsed:", parsedprogram, "\n"


  ra = parsedprogram.toRA()
  generateDot(ra,headname+'.dot')
  print "ra:", ra, "\n"

  physicalplan = optimize(ra, target=CCAlgebra, source=LogicalAlgebra)

  LOG.info("physical: %s",physicalplan[0])
  LOG.info('args=%s',physicalplan[0][1].args)
  LOG.info('joinconditions=%s',physicalplan[0][1].joinconditions)
  LOG.info('leftconditions=%s',physicalplan[0][1].leftconditions)
  LOG.info('rightconditions=%s',physicalplan[0][1].rightconditions)
  LOG.info('final=%s',physicalplan[0][1].finalcondition)
  tmp = sc.cpp_code(physicalplan,headname,dis=set_sem)
  LOG.info('cpp_code obj:%s',tmp)
  tmp.gen_code()

