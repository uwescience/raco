import abc
from raco import algebra
from raco import expression
from raco import catalog
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

    
  def generateDefinition(self):
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
    self.unifiedTupleType = StagedTupleRef(gensym(), self.scheme())
    state.addDeclarations([self.unifiedTupleType.generateDefinition()])

    self.right.produce(state)
    self.left.produce(state)

  def consume(self, t, src, state):
    union_template = """auto %(unified_tuple_name)s = transpose<%(unified_tuple_typename)s>(%(src_tuple_name)s);
                        %(inner_plan_compiled)s"""

    unified_tuple_typename = self.unifiedTupleType.getTupleTypename()
    unified_tuple_name = self.unifiedTupleType.name
    src_tuple_name = t.name

    inner_plan_compiled = self.parent.consume(self.unifiedTupleType, self, state)
    return union_template % locals()


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
    state.addDeclarations( [newtuple.generateDefinition()] )

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


from algebra import ZeroaryOperator
class CFileScan(algebra.Scan):

    @abc.abstractmethod
    def __get_ascii_scan_template__(self):
       return

    @abc.abstractmethod
    def __get_binary_scan_template__(self):
        return

    def compileme(self, resultsym):
        # TODO use the identifiers (don't split str and extract)
        #name = self.relation_key
        LOG.debug('compiling file scan for relation_key %s' % self.relation_key)
        name = str(self.relation_key).split(':')[2]

        #tup = (resultsym, self.originalterm.originalorder, self.originalterm)
        #self.trace("// Original query position of %s: term %s (%s)" % tup)

        if isinstance(self.relation_key, catalog.ASCIIFile):
            code = self.__get_ascii_scan_template__() % locals()
        else:
            code = self.__get_binary_scan_template__() % locals()
        return code

    def __str__(self):
        return "%s(%s)" % (self.opname(), self.relation_key)

    def __eq__(self, other):
        """
        For what we are using FileScan for, the only use
        of __eq__ is in hashtable lookups for CSE optimization.
        We omit self.schema because the relation_key determines
        the level of equality needed.

        This could break other things, so better may be to
        make a normalized copy of an expression. This could
        include simplification but in the case of Scans make
        the scheme more generic.

        @see MemoryScan.__eq__
        """
        return ZeroaryOperator.__eq__(self, other) and \
               self.relation_key == other.relation_key
