from raco.datalog.grammar import parse
from raco.scheme import Scheme
from raco.catalog import ASCIIFile
from raco.language import PythonAlgebra, PseudoCodeAlgebra, CCAlgebra#, ProtobufAlgebra
from raco.algebra import LogicalAlgebra
from raco.compile import compile, optimize, common_subexpression_elimination, showids
import scan_code_ver2 as sc
import raco.boolean
from generateDot import generateDot

#query = 'A(a1) :- R(a1,x),S(x,y),T(y,z),U(z,a1),100=z,y=50'
query = 'Triangle(x,y,z) :- R(x,y),S(y,z),T(z,x)'
#query = 'Triangle(x,y,z) :- edges(x,y),edges(y,z),edges(z,x),x<y,y<z'
#query = 'California(x,z) :- edges1(x,y1),edges1(y1,y2),edges2(y2,z)'
#query = 'A(x,z) :- edges(x,y1),edges(y1,y2),edges(y2,z)'
#query = 'we(a,c) :- edges(a,b),edges(b,c)'

print "query:", query, "\n"


parsedprogram = parse(query)
print "parsed:", parsedprogram, "\n"


ra = parsedprogram.toRA()
generateDot(ra,'diamond.dot')
print "ra:", ra, "\n"

physicalplan = optimize(ra, target=CCAlgebra, source=LogicalAlgebra)
#print "physical plan:", physicalplan
print 'args=',physicalplan[0][1].args
print 'joinconditions=',physicalplan[0][1].joinconditions
print 'leftconditions=',physicalplan[0][1].leftconditions
print 'rightconditions=',physicalplan[0][1].rightconditions
print 'final=',physicalplan[0][1].finalcondition
#tmp = sc.cpp_code(physicalplan,'california')
tmp = sc.cpp_code(physicalplan,'triangle_nosel')
print 'cpp_code result:',tmp
tmp.gen_code()

