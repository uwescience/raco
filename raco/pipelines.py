import abc
from raco.utility import emitlist

import logging
LOG = logging.getLogger(__name__)

# for testing output of queries
class TestEmit:
  def __init__(self, lang):
    self.language = lang
  def consume(self,t,src,state):
    return self.language.log_unquoted("%s" % t.name)

class CompileState:

    def __init__(self, cse=True):
        self.declarations = []
        self.pipelines = []
        self.initializers = []

        # { expression => symbol for materialized result }
        self.materialized = {}

        # { symbol => tuple type definition }
        self.tupledefs = {}

        self.common_subexpression_elim = cse

    def addDeclarations(self, d):
        self.declarations += d

    def addInitializers(self, i):
        self.initializers += i

    def addPipeline(self, p):
        self.pipelines.append(p)

    def addCode(self, c):
        self.pipelines.append(c)

    def getInitCode(self):
        return emitlist(self.initializers)

    def getDeclCode(self):
        # declarations is a set
        # If this ever becomes a bottleneck when declarations are strings,
        # as in clang, then resort to at least symbol name deduping.
        s = set()
        def f(x):
            if x in s: return False
            else:
                s.add(x)
                return True

        # keep in original order
        return emitlist(filter(f, self.declarations))

    def getExecutionCode(self):
        return emitlist(self.pipelines)

    def lookupExpr(self, expr):
        if self.common_subexpression_elim:
            return self.materialized.get(expr)
        else:
            # if CSE is turned off then always return None for expression matches
            return None

    def saveExpr(self, sym, expr):
        self.materialized[expr] = sym

    def lookupTupleDef(self, sym):
        return self.tupledefs.get(sym)

    def saveTupleDef(self, sym, tupledef):
        self.tupledefs[sym] = tupledef

  
class Pipelined(object):
    '''
    Trait to provide the compilePipeline method
    for calling into pipeline style compilation.
    '''
  
    __metaclass__ = abc.ABCMeta

    def __markAllParents__(self):
      root = self
      
      def markChildParent(op):
        for c in op.children():
          c.parent = op
        return []
          
      [_ for _ in root.postorder(markChildParent)]
      root.parent = TestEmit(root.language)
      
    @abc.abstractmethod
    def produce(self, state):
      """Denotation for producing a tuple"""
      return
    
    @abc.abstractmethod
    def consume(self, inputTuple, fromOp, state):
      """Denotation for consuming a tuple"""
      return

    def compilePipeline(self, resultsym):
      self.__markAllParents__()

      state = CompileState()
      
      # TODO bound
      # TODO should be using resultsym?? for what here?

      state.addCode( self.language.comment("Compiled subplan for %s" % self) )

      self.produce(state)

      state.addCode( self.language.log("Evaluating subplan %s" % self) )

      return state
