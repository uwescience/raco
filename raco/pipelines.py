import abc
from raco.utility import emitlist

# for testing output of queries
class TestEmit:
  def __init__(self, lang):
    self.language = lang
  def consume(self,t,src,state):
    return self.language.log_unquoted("%s" % t.name)

class CompileState:

    def __init__(self):
        self.declarations = []
        self.pipelines = []
        self.initializers = []

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
        return emitlist(self.declarations)

    def getExecutionCode(self):
        return emitlist(self.pipelines)

  
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
