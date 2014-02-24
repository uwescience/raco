# TODO: To be refactored into shared memory lang,
# where you plugin in the sequential shared memory language specific codegen

from raco import algebra
from raco import expression
from raco import catalog
from raco.language import Language
from raco import rules
from raco.pipelines import Pipelined
from raco.clangcommon import StagedTupleRef
from raco import clangcommon

from algebra import gensym

import logging
LOG = logging.getLogger(__name__)

import os.path


template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_templates")

def readtemplate(fname):
    return file(os.path.join(template_path, fname)).read()

template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_templates")

base_template = readtemplate("base_query.template")
twopass_select_template = readtemplate("precount_select.template")
hashjoin_template = readtemplate("hashjoin.template")
filteringhashjoin_template = ""
filtering_nestedloop_join_chain_template = ""#readtemplate("filtering_nestedloop_join_chain.template")
ascii_scan_template = readtemplate("ascii_scan.template")
binary_scan_template = readtemplate("binary_scan.template")



class CStagedTupleRef(StagedTupleRef):
  def __additionalDefinitionCode__(self):
    constructor_template = """
    public:
    %(tupletypename)s (relationInfo * rel, int row) {
      %(copies)s
    }
    """

    copytemplate = """_fields[%(fieldnum)s] = rel->relation[row*rel->fields + %(fieldnum)s];
    """

    copies = ""
    # TODO: actually list the trimmed schema offsets
    for i in range(0, len(self.scheme)):
      fieldnum = i
      copies += copytemplate % locals()
      
    tupletypename = self.getTupleTypename()
    return constructor_template % locals()
    

class CC(Language):
    @classmethod
    def new_relation_assignment(cls, rvar, val):
        return """
    %s
    %s
    """ % (cls.relation_decl(rvar), cls.assignment(rvar, val))

    @classmethod
    def relation_decl(cls, rvar):
        return "struct relationInfo *%s;" % rvar

    @classmethod
    def assignment(cls, x, y):
        return "%s = %s;" % (x, y)

    @staticmethod
    def initialize(resultsym):
        return ""
      
    @staticmethod
    def body(compileResult, resultsym):
      queryexec = compileResult.getExecutionCode()
      initialized = compileResult.getInitCode()
      declarations = compileResult.getDeclCode()
      return base_template % locals()

    @staticmethod
    def finalize(resultsym):
        return ""

    @staticmethod
    def log(txt):
        return  """std::cout << "%s" << std::endl;
        """ % txt
      
    @staticmethod
    def log_unquoted(code):
      return """std::cout << %s << std::endl;
      """ % code

    @staticmethod
    def comment(txt):
        return  "// %s\n" % txt

    nextstrid = 0
    @classmethod
    def newstringident(cls):
        r = """str_%s""" % (cls.nextstrid)
        cls.nextstrid += 1
        return r

    @classmethod
    def compile_numericliteral(cls, value):
        return '%s'%(value), []

    @classmethod
    def compile_stringliteral(cls, s):
        sid = cls.newstringident()
        init = """auto %s = string_index.string_lookup("%s");""" % (sid, s)
        return """(%s)""" % sid, [init]
        #raise ValueError("String Literals not supported in C language: %s" % s)

    @classmethod
    def negation(cls, input):
        innerexpr, inits = input
        return "(!%s)" % (innerexpr,), inits

    @classmethod
    def boolean_combine(cls, args, operator="&&"):
        opstr = " %s " % operator
        conjunc = opstr.join(["(%s)" % arg for arg, _ in args])
        inits = reduce(lambda sofar, x: sofar+x, [d for _, d in args])
        LOG.debug("conjunc: %s", conjunc)
        return "( %s )" % conjunc, inits

    @classmethod
    def compile_attribute(cls, expr):
        if isinstance(expr, expression.NamedAttributeRef):
            raise TypeError("Error compiling attribute reference %s. C compiler only support unnamed perspective.  Use helper function unnamed." % expr)
        if isinstance(expr, expression.UnnamedAttributeRef):
            symbol = expr.tupleref.name
            position = expr.position # NOTE: this will only work in Selects right now
            return '%s.get(%s)' % (symbol, position), []

class CCOperator (Pipelined):
    language = CC
    
class MemoryScan(algebra.Scan, CCOperator):
  def produce(self, state):
    code = ""
    #generate the materialization from file into memory
    #TODO split the file scan apart from this in the physical plan

    fs = FileScan(self.relation_key, self._scheme)

    # Common subexpression elimination
    # don't scan the same file twice
    resultsym = state.lookupExpr(fs)
    LOG.debug("lookup %s(h=%s) => %s", fs, fs.__hash__(), resultsym)
    if not resultsym:
        #TODO for now this will break whatever relies on self.bound like reusescans
        #Scan is the only place where a relation is declared
        resultsym = gensym()

        code += fs.compileme(resultsym)
        state.saveExpr(resultsym, fs)


    # now generate the scan from memory
    inputsym = resultsym

    #TODO: generate row variable to avoid naming conflict for nested scans
    memory_scan_template = """for (uint64_t i : %(inputsym)s->range()) {
          %(tuple_type)s %(tuple_name)s(%(inputsym)s, i);
          
          %(inner_plan_compiled)s
       } // end scan over %(inputsym)s
       """

    stagedTuple = state.lookupTupleDef(inputsym)
    if not stagedTuple:
        # if the tuple type definition does not yet exist, then
        # create it and add its definition
        stagedTuple = CStagedTupleRef(inputsym, self.scheme())
        state.saveTupleDef(inputsym, stagedTuple)

        tuple_type_def = stagedTuple.generateDefition()
        state.addDeclarations([tuple_type_def])


    tuple_type = stagedTuple.getTupleTypename()
    tuple_name = stagedTuple.name
    
    inner_plan_compiled = self.parent.consume(stagedTuple, self, state)

    code += memory_scan_template % locals()
    state.addPipeline(code)


  def consume(self, t, src, state):
    assert False, "as a source, no need for consume"


  def __eq__(self, other):
    """
    For what we are using MemoryScan for, the only use
    of __eq__ is in hashtable lookups for CSE optimization.
    We omit self.schema because the relation_key determines
    the level of equality needed.

    @see FileScan.__eq__
    """
    return ZeroaryOperator.__eq__(self, other) and \
           self.relation_key == other.relation_key



from algebra import ZeroaryOperator
class FileScan(algebra.Scan):

    def compileme(self, resultsym):
        # TODO use the identifiers (don't split str and extract)
        #name = self.relation_key
        name = str(self.relation_key).split(':')[2]

        #tup = (resultsym, self.originalterm.originalorder, self.originalterm)
        #self.trace("// Original query position of %s: term %s (%s)" % tup)

        if isinstance(self.relation_key, catalog.ASCIIFile):
            code = ascii_scan_template % locals()
        else:
            code = binary_scan_template % locals()
        return code
      
    def __str__(self):
      return "%s(%s)" % (self.opname(), self.relation_key)

    def __eq__(self, other):
        """
        For what we are using FileScan for, the only use
        of __eq__ is in hashtable lookups for CSE optimization.
        We omit self.schema because the relation_key determines
        the level of equality needed.

        @see MemoryScan.__eq__
        """
        return ZeroaryOperator.__eq__(self, other) and \
               self.relation_key == other.relation_key


class HashJoin(algebra.Join, CCOperator):
  _i = 0

  @classmethod
  def __genHashName__(cls):
    name = "hash_%03d" % cls._i;
    cls._i += 1
    return name
  
  def produce(self, state):
    if not isinstance(self.condition, expression.EQ):
      msg = "The C compiler can only handle equi-join conditions of a single attribute: %s" % self.condition
      raise ValueError(msg)
    
    #self._hashname = self.__genHashName__()
    #self.outTuple = CStagedTupleRef(gensym(), self.scheme())
    
    self.right.childtag = "right"
    self.right.produce(state)
    
    self.left.childtag = "left"
    self.left.produce(state)

  
  def consume(self, t, src, state):
    if src.childtag == "right":
      declr_template =  """std::unordered_map<int64_t, std::vector<%(in_tuple_type)s>* > %(hashname)s;
      """
      
      right_template = """insert(%(hashname)s, %(keyname)s, %(keypos)s);
      """   
      
      # FIXME generating this here is an ugly hack to fix the mutable problem
      self._hashname = self.__genHashName__()
      LOG.debug("generate hashname %s for %s", self, self._hashname)
      # FIXME generating this here is another ugly hack to fix the mutable problem
      self.outTuple = CStagedTupleRef(gensym(), self.scheme())

      hashname = self._hashname
      keyname = t.name
      keypos = self.condition.right.position-len(self.left.scheme())
      
      out_tuple_type_def = self.outTuple.generateDefition()
      self.rightTuple = t #TODO: this induces a right->left dependency
      in_tuple_type = self.rightTuple.getTupleTypename()

      # declaration of hash map
      hashdeclr =  declr_template % locals()
      state.addDeclarations([out_tuple_type_def,hashdeclr])
      
      # materialization point
      code = right_template % locals()
      
      return code
    
    if src.childtag == "left":
      left_template = """
      for (auto %(right_tuple_name)s : lookup(%(hashname)s, %(keyname)s.get(%(keypos)s))) {
        %(out_tuple_type)s %(out_tuple_name)s = combine<%(out_tuple_type)s, %(keytype)s, %(right_tuple_type)s> (%(keyname)s, %(right_tuple_name)s);
     %(inner_plan_compiled)s 
  }
  """
      hashname = self._hashname
      keyname = t.name
      keytype = t.getTupleTypename()
      keypos = self.condition.left.position
      
      right_tuple_type = self.rightTuple.getTupleTypename()
      right_tuple_name = self.rightTuple.name

      # or could make up another name
      #right_tuple_name = CStagedTupleRef.genname() 

      out_tuple_type = self.outTuple.getTupleTypename()
      out_tuple_name =self.outTuple.name
      
      inner_plan_compiled = self.parent.consume(self.outTuple, self, state)
      
      code = left_template % locals()
      return code

    assert False, "src not equal to left or right"
      

def indentby(code, level):
    indent = " " * ((level + 1) * 6)
    return "\n".join([indent + line for line in code.split("\n")])





# iteration  over table + insertion into hash table with filter

class CUnionAll(clangcommon.CUnionAll, CCOperator): pass

class CApply(clangcommon.CApply, CCOperator): pass
  
class CProject(clangcommon.CProject, CCOperator): pass

class CSelect(clangcommon.CSelect, CCOperator): pass
    

class CCAlgebra(object):
    language = CC

    operators = [
    #TwoPassHashJoin,
    #FilteringNestedLoopJoin,
    #TwoPassSelect,
    #FileScan,
    MemoryScan,
    CSelect,
    CUnionAll,
    CApply,
    CProject,
    HashJoin
  ]
    rules = [
     #rules.OneToOne(algebra.Join,TwoPassHashJoin),
    #rules.removeProject(),
    rules.CrossProduct2Join(),
#    FilteringNestedLoopJoinRule(),
#    FilteringHashJoinChainRule(),
#    LeftDeepFilteringJoinChainRule(),
    rules.OneToOne(algebra.Select,CSelect),
 #   rules.OneToOne(algebra.Select,TwoPassSelect),
    rules.OneToOne(algebra.Scan,MemoryScan),
    rules.OneToOne(algebra.Apply, CApply),
    rules.OneToOne(algebra.Join,HashJoin),
    rules.OneToOne(algebra.Project, CProject),
    rules.OneToOne(algebra.Union,CUnionAll) #TODO: obviously breaks semantics
  #  rules.FreeMemory()
  ]
