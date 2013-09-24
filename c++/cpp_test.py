from raco.datalog.grammar import parse
from raco.scheme import Scheme
from raco.catalog import ASCIIFile
from raco.language import PythonAlgebra, PseudoCodeAlgebra, CCAlgebra#, ProtobufAlgebra
from raco.algebra import LogicalAlgebra
from raco.compile import compile, optimize, common_subexpression_elimination, showids
import scan_code_ver2 as sc
import raco.boolean
from generateDot import generateDot

import sys
if len(sys.argv) > 1:
    query = sys.argv[1]
    set_sem = bool(sys.argv[2])
    if (set_sem=='True'): set_sem = True
    else: set_sem=False
else:
    #query = 'Long(a1) :- R(a1,x),S(x,y),T(y,z),U(z,w),V(w,a1),100=z,y=50'
    #query = 'Triangle(x,y,z) :- R(x,y),S(y,z),T(z,x)'
    #query = 'Triangle(x,y,z) :- edges(x,y),edges(y,z),edges(z,x),x<y,y<z'
    #query = 'California(x,z) :- edges1(x,y1),edges1(y1,y2),edges2(y2,z)'
    query = 'fof(a,c) :- edges(a,b),edges(b,c)'
    #query = 'fofsel(a,c) :- edges(a,b),edges(b,c),b<c'
    #query = 'fofdir(a,c) :- edges(a,b),edges(b,c),a<b,b<c'
    #query = 'mutual(a,b) :- edges(a,b),edges(b,a)'

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
#print "physical plan:", physicalplan
print 'args=',physicalplan[0][1].args
print 'joinconditions=',physicalplan[0][1].joinconditions
print 'leftconditions=',physicalplan[0][1].leftconditions
print 'rightconditions=',physicalplan[0][1].rightconditions
print 'final=',physicalplan[0][1].finalcondition
tmp = sc.cpp_code(physicalplan,headname,dis=set_sem)  #dis=True
print 'cpp_code result:',tmp
tmp.gen_code()

