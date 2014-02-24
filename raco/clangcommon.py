from raco import algebra
from raco import expression
from algebra import gensym
from expression.expression import UnnamedAttributeRef

import logging
LOG = logging.getLogger(__name__)

# TODO:
# The following is actually a staged materialized tuple ref.
# we should also add a staged reference tuple ref that just has relationsymbol and row  
class StagedTupleRef:
  nextid = 0
  
  @classmethod
  def genname(cls):
    # use StagedTupleRef so everyone shares one mutable copy of nextid
    x = StagedTupleRef.nextid   
    StagedTupleRef.nextid+=1
    return "t_%03d" % x
  
  def __init__(self, relsym, scheme):
    self.name = self.genname()
    self.relsym = relsym
    self.scheme = scheme
    self.__typename = None
  
  def getTupleTypename(self):
    if self.__typename==None:
      fields = ""
      relsym = self.relsym
      for i in range(0, len(self.scheme)):
        fieldnum = i
        fields += "_%(fieldnum)s" % locals()
        
      self.__typename = "MaterializedTupleRef_%(relsym)s%(fields)s" % locals()
    
    return self.__typename

    
  def generateDefition(self):
    fielddeftemplate = """int64_t _fields[%(numfields)s];
    """
    template = """
          // can be just the necessary schema
  class %(tupletypename)s {
    private:
    %(fielddefs)s
    
    public:
    int64_t get(int field) const {
      return _fields[field];
    }
    
    void set(int field, int64_t val) {
      _fields[field] = val;
    }
    
    int numFields() const {
      return %(numfields)s;
    }
    
    %(tupletypename)s () {
      // no-op
    }

    %(tupletypename)s (std::vector<int64_t> vals) {
      for (int i=0; i<vals.size(); i++) _fields[i] = vals[i];
    }
    
    std::ostream& dump(std::ostream& o) const {
      o << "Materialized(";
      for (int i=0; i<numFields(); i++) {
        o << _fields[i] << ",";
      }
      o << ")";
      return o;
    }
    
    %(additional_code)s
  } %(after_def_code)s;
  std::ostream& operator<< (std::ostream& o, const %(tupletypename)s& t) {
    return t.dump(o);
  }

  """
    getcases = ""
    setcases = ""
    copies = ""
    numfields = len(self.scheme)
    fielddefs = fielddeftemplate % locals()

    
    additional_code = self.__additionalDefinitionCode__()
    after_def_code = self.__afterDefinitionCode__()


    tupletypename = self.getTupleTypename()
    relsym = self.relsym
      
    code = template % locals()
    return code
  
  def __additionalDefinitionCode__(self):
    return ""

  def __afterDefinitionCode__(self):
      return ""
  

def getTaggingFunc(t):
  """ 
  Return a visitor function that will tag 
  UnnamedAttributes with the provided TupleRef
  """

  def tagAttributes(expr):
    # TODO non mutable would be nice
    if isinstance(expr, expression.UnnamedAttributeRef):
      expr.tupleref = t

    return None
  
  return tagAttributes

 
class CSelect(algebra.Select):
  def produce(self, state):
    self.input.produce(state)
    
  def consume(self, t, src, state):
    basic_select_template = """if (%(conditioncode)s) {
      %(inner_code_compiled)s
    }
    """

    # tag the attributes with references
    # TODO: use an immutable approach instead (ie an expression Visitor for compiling)
    [_ for _ in self.condition.postorder(getTaggingFunc(t))]
    
    # compile the predicate into code
    conditioncode, cond_inits = self.language.compile_boolean(self.condition)
    state.addInitializers(cond_inits)
    
    inner_code_compiled = self.parent.consume(t, self, state)
    
    code = basic_select_template % locals()
    return code
  
  
class CUnionAll(algebra.Union):
  def produce(self, state):
    self.right.produce(state)
    self.left.produce(state)

  def consume(self, t, src, state):
    #FIXME: expect a bug: because we have not forced
    #CCOperators to be immutable (e.g. if self.parent is a HashJoin), then this is problematic
    # For now HashJoin is just lucky
    return self.parent.consume(t, self, state)


class CApply(algebra.Apply):
  def produce(self, state):
    self.input.produce(state)
  
  def consume(self, t, src, state):
    return self.parent.consume(t, self, state)


class CProject(algebra.Project):
  def produce(self, state):
    self.input.produce(state)
  
  def consume(self, t, src, state):
    code = ""

    # always does an assignment to new tuple
    newtuple = StagedTupleRef(gensym(), self.scheme())
    state.addDeclarations( [newtuple.generateDefition()] )
    
    assignment_template = """%(dst_name)s.set(%(dst_fieldnum)s, %(src_name)s.get(%(src_fieldnum)s));
    """
    
    dst_name = newtuple.name
    dst_type_name = newtuple.getTupleTypename()
    src_name = t.name

    # declaration of tuple instance
    code += """%(dst_type_name)s %(dst_name)s;
    """ % locals()
    
    for dst_fieldnum, src_expr in enumerate(self.columnlist):
      if isinstance(src_expr, UnnamedAttributeRef):
        src_fieldnum = src_expr.position
      else:
        assert False, "Unsupported Project expression"
      code += assignment_template % locals()
      
    innercode = self.parent.consume(newtuple, self, state)
    code+=innercode
      
    return code