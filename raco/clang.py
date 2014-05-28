# TODO: make it pass with flake8 test
# flake8: noqa

# TODO: To be refactored into shared memory lang,
# where you plugin in the sequential shared memory language specific codegen

from raco import algebra
from raco import expression
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
    def pipeline_wrap(ident, code, attrs):
        # TODO: timer, etc
        return code

    @staticmethod
    def group_wrap(ident, code, attrs):
        # TODO: timer, etc
        return code

    @staticmethod
    def log(txt):
        return  """std::cout << "%s" << std::endl;
        """ % txt
      
    @staticmethod
    def log_unquoted(code, level=0):
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
        return '%s'%(value), [], []

    @classmethod
    def compile_stringliteral(cls, s):
        sid = cls.newstringident()
        init = """auto %s = string_index.string_lookup(%s);""" % (sid, s)
        return """(%s)""" % sid, [], [init]
        #raise ValueError("String Literals not supported in C language: %s" % s)

    @classmethod
    def negation(cls, input):
        innerexpr, inits = input
        return "(!%s)" % (innerexpr,), [], inits

    @classmethod
    def negative(cls, input):
        innerexpr, decls, inits = input
        return "(-%s)" % (innerexpr,), decls, inits

    @classmethod
    def expression_combine(cls, args, operator="&&"):
        opstr = " %s " % operator
        conjunc = opstr.join(["(%s)" % arg for arg, _, _ in args])
        decls = reduce(lambda sofar, x: sofar+x, [d for _, d, _ in args])
        inits = reduce(lambda sofar, x: sofar+x, [d for _, _, d in args])
        LOG.debug("conjunc: %s", conjunc)
        return "( %s )" % conjunc, decls, inits

    @classmethod
    def compile_attribute(cls, expr):
        if isinstance(expr, expression.NamedAttributeRef):
            raise TypeError("Error compiling attribute reference %s. C compiler only support unnamed perspective.  Use helper function unnamed." % expr)
        if isinstance(expr, expression.UnnamedAttributeRef):
            symbol = expr.tupleref.name
            position = expr.position # NOTE: this will only work in Selects right now
            assert position >= 0
            return '%s.get(%s)' % (symbol, position), [], []

class CCOperator (Pipelined):
    language = CC

    def new_tuple_ref(self, sym, scheme):
        return CStagedTupleRef(sym, scheme)

from algebra import UnaryOperator
class MemoryScan(algebra.UnaryOperator, CCOperator):

  def produce(self, state):
      self.input.produce(state)

  # TODO: when have pipeline tree representation, will have a consumeMaterialized() method instead;
  # for now we reuse the tuple-based consume
  def consume(self, inputsym, src, state):

    # now generate the scan from memory

    #TODO: generate row variable to avoid naming conflict for nested scans
    memory_scan_template = """for (uint64_t i : %(inputsym)s->range()) {
          %(tuple_type)s %(tuple_name)s(%(inputsym)s, i);
          
          %(inner_plan_compiled)s
       } // end scan over %(inputsym)s
       """

    stagedTuple = state.lookupTupleDef(inputsym)
    tuple_type = stagedTuple.getTupleTypename()
    tuple_name = stagedTuple.name
    
    inner_plan_compiled = self.parent.consume(stagedTuple, self, state)

    code = memory_scan_template % locals()
    state.setPipelineProperty("type", "in_memory")
    state.addPipeline(code)
    return None


  def shortStr(self):
    return "%s" % (self.opname())


  def __eq__(self, other):
    """
    For what we are using MemoryScan for, the only use
    of __eq__ is in hashtable lookups for CSE optimization.
    We omit self.schema because the relation_key determines
    the level of equality needed.

    @see FileScan.__eq__
    """
    return UnaryOperator.__eq__(self, other)


class CGroupBy(algebra.GroupBy, CCOperator):
    _i = 0

    @classmethod
    def __genHashName__(cls):
        name = "group_hash_%03d" % cls._i;
        cls._i += 1
        return name

    def produce(self, state):
        assert len(self.grouping_list) <= 1, \
            "%s does not currently support groupings of more than 1 attribute" % self.__class__.__name__
        assert len(self.aggregate_list) == 1, \
            "%s currently only supports aggregates of 1 attribute" % self.__class__.__name__

        self.useMap = len(self.grouping_list) > 0

        if self.useMap:
            declr_template = """std::unordered_map<int64_t, int64_t> %(hashname)s;
      """
        else:
            declr_template = """int64_t %(hashname)s;
            """

        self.hashname = self.__genHashName__()
        hashname = self.hashname


        hash_declr = declr_template % locals()
        state.addDeclarations([hash_declr])

        LOG.debug("aggregates: %s", self.aggregate_list)
        LOG.debug("columns: %s", self.column_list)
        LOG.debug("groupings: %s", self.grouping_list)
        LOG.debug("groupby scheme: %s", self.scheme())
        LOG.debug("groupby scheme[0] type: %s", type(self.scheme()[0]))

        self.input.produce(state)

        # now that everything is aggregated, produce the tuples
        assert (not self.useMap) or isinstance(self.column_list[0], expression.UnnamedAttributeRef), \
            "assumes first column is the key and second is aggregate result"

        if self.useMap:
            produce_template = """for (auto it=%(hashname)s.begin(); it!=%(hashname)s.end(); it++) {
            %(output_tuple_type)s %(output_tuple_name)s({it->first, it->second});
            %(inner_code)s
            }
            """
        else:
            produce_template = """{
            %(output_tuple_type)s %(output_tuple_name)s({ %(hashname)s });
            %(inner_code)s
            }
            """

        output_tuple = CStagedTupleRef(gensym(), self.scheme())
        output_tuple_name = output_tuple.name
        output_tuple_type = output_tuple.getTupleTypename()
        state.addDeclarations([output_tuple.generateDefinition()])

        inner_code = self.parent.consume(output_tuple, self, state)
        code = produce_template % locals()
        state.setPipelineProperty("type", "in_memory")
        state.addPipeline(code)

    def consume(self, inputTuple, fromOp, state):
        if self.useMap:
            materialize_template = """%(op)s_insert(%(hashname)s, %(tuple_name)s, %(keypos)s, %(valpos)s);
      """
        else:
            materialize_template = """%(op)s_insert(%(hashname)s, %(tuple_name)s, %(valpos)s);
            """

        hashname = self.hashname
        tuple_name = inputTuple.name

        # make key from grouped attributes
        if self.useMap:
            keypos = self.grouping_list[0].get_position(self.scheme())

        # get value positions from aggregated attributes
        valpos = self.aggregate_list[0].input.get_position(self.scheme())

        op = self.aggregate_list[0].__class__.__name__

        code = materialize_template % locals()
        return code


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

    # find the attribute that corresponds to the right child
    self.rightCondIsRightAttr = \
      self.condition.right.position >= len(self.left.scheme())
    self.leftCondIsRightAttr = \
      self.condition.left.position >= len(self.left.scheme())
    assert self.rightCondIsRightAttr ^ self.leftCondIsRightAttr

    self.right.childtag = "right"
    hashsym = state.lookupExpr(self.right)

    if not hashsym:
        # if right child never bound then store hashtable symbol and
        # call right child produce
        self._hashname = self.__genHashName__()
        LOG.debug("generate hashname %s for %s", self._hashname, self)
        state.saveExpr(self.right, self._hashname)
        self.right.produce(state)
    else:
        # if found a common subexpression on right child then
        # use the same hashtable
        self._hashname = hashsym
        LOG.debug("reuse hash %s for %s", self._hashname, self)

    self.left.childtag = "left"
    self.left.produce(state)

  
  def consume(self, t, src, state):
    if src.childtag == "right":
      declr_template =  """std::unordered_map<int64_t, std::vector<%(in_tuple_type)s>* > %(hashname)s;
      """
      
      right_template = """insert(%(hashname)s, %(keyname)s, %(keypos)s);
      """   

      hashname = self._hashname
      keyname = t.name

      # find the attribute that corresponds to the right child
      if self.rightCondIsRightAttr:
        keypos = self.condition.right.position-len(self.left.scheme())
      else:
        keypos = self.condition.left.position-len(self.left.scheme())


      in_tuple_type = t.getTupleTypename()

      # declaration of hash map
      hashdeclr =  declr_template % locals()
      state.addDeclarations([hashdeclr])
      
      # materialization point
      code = right_template % locals()
      
      return code
    
    if src.childtag == "left":
      left_template = """
      for (auto %(right_tuple_name)s : lookup(%(hashname)s, %(keyname)s.get(%(keypos)s))) {
        auto %(out_tuple_name)s = combine<%(out_tuple_type)s> (%(keyname)s, %(right_tuple_name)s);
     %(inner_plan_compiled)s 
  }
  """
      hashname = self._hashname
      keyname = t.name
      keytype = t.getTupleTypename()

      if self.rightCondIsRightAttr:
          keypos = self.condition.left.position
      else:
          keypos = self.condition.right.position

      right_tuple_name = gensym()

      outTuple = CStagedTupleRef(gensym(), self.scheme())
      out_tuple_type_def = outTuple.generateDefinition()
      out_tuple_type = outTuple.getTupleTypename()
      out_tuple_name = outTuple.name

      state.addDeclarations([out_tuple_type_def])

      inner_plan_compiled = self.parent.consume(outTuple, self, state)
      
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

class CFileScan(clangcommon.CFileScan, CCOperator):
    def __get_ascii_scan_template__(self): return ascii_scan_template

    def __get_binary_scan_template__(self): return binary_scan_template



class MemoryScanOfFileScan(rules.Rule):
    """A rewrite rule for making a scan into materialization in memory then memory scan"""
    def fire(self, expr):
        if isinstance(expr, algebra.Scan) and not isinstance(expr, CFileScan):
            return MemoryScan(CFileScan(expr.relation_key, expr.scheme()))
        return expr

    def __str__(self):
        return "Scan => MemoryScan(FileScan)"


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
    CGroupBy,
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
  #  rules.OneToOne(algebra.Scan,MemoryScan),
    MemoryScanOfFileScan(),
    rules.OneToOne(algebra.Apply, CApply),
    rules.OneToOne(algebra.Join,HashJoin),
    rules.OneToOne(algebra.GroupBy,CGroupBy),
    rules.OneToOne(algebra.Project, CProject),
    rules.OneToOne(algebra.Union,CUnionAll) #TODO: obviously breaks semantics
  #  rules.FreeMemory()
  ]
