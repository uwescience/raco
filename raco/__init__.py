from raco.datalog.grammar import parse
from raco.scheme import Scheme
from raco.language import PythonAlgebra, PseudoCodeAlgebra, CCAlgebra#, ProtobufAlgebra
from raco.language import MyriaAlgebra
from raco.algebra import LogicalAlgebra
from raco.compile import compile, optimize, common_subexpression_elimination, showids
from utility import emit

import raco.algebra

class RACompiler:
  """Thin wrapper interface for lower level functions parse, optimize, compile"""
  
  def fromDatalog(self, program):
    """Parse datalog and convert to RA"""
    self.physicalplan = None
    self.target = None
    self.source = program
    self.parsed = parse(program)
    self.logicalplan = self.parsed.toRA()

  def optimize(self, target=MyriaAlgebra, eliminate_common_subexpressions=False):
    """Convert logical plan to physical plan"""
    self.target = target
    self.physicalplan = optimize(
           self.logicalplan
         , target = self.target
         , source = LogicalAlgebra
         , eliminate_common_subexpressions=eliminate_common_subexpressions
    )

  def compile(self):
    """Compile physical plan to linearized form for execution"""
    #TODO: Fix this
    algebra.reset()
    exprs = self.physicalplan
    lang = self.target.language
    exprcode = []
    exprcode.append(lang.preamble(query=self.source, plan=self.logicalplan))
    for result, expr in exprs:
      init = lang.initialize(result)
      body = expr.compile(result)
      final = lang.finalize(result)
      exprcode.append(emit(init, body, final))
    exprcode.append(lang.postamble(query=self.source, plan=self.logicalplan))
    return  emit(*exprcode)

