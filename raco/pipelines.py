import abc
    

# for testing output of queries
class TestEmit:
  def __init__(self, lang):
    self.language = lang
  def consume(self,t,src):
    return self.language.log_unquoted("%s" % t.name), [], []
  
  
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
    def produce(self):
      """Denotation for producing a tuple"""
      return
    
    @abc.abstractmethod
    def consume(self, inputTuple, fromOp):
      """Denotation for consuming a tuple"""
      return

    def compilePipeline(self, resultsym):
      self.__markAllParents__()
      
      # TODO bound
      # TODO should be using resultsym?? for what here?

      code = self.language.comment("Compiled subplan for %s" % self)
      selfcode, selfdecls, selfinits = self.produce()

      code += self.language.log("Evaluating subplan %s" % self)
      code += selfcode
      
      return (code, selfdecls, selfinits)
