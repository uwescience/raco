# TODO: To be refactored into shared memory lang,
# where you plugin in the sequential shared memory language specific codegen

from raco import algebra
from raco import expression
from raco import catalog
from raco.language import Language
from raco import rules
from raco.utility import emitlist
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
initialize, querydef_init, finalize = base_template.split("// SPLIT ME HERE")
twopass_select_template = readtemplate("precount_select.template")
hashjoin_template = readtemplate("hashjoin.template")
filteringhashjoin_template = ""
filtering_nestedloop_join_chain_template = ""#readtemplate("filtering_nestedloop_join_chain.template")
ascii_scan_template = readtemplate("ascii_scan.template")
binary_scan_template = readtemplate("binary_scan.template")

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
        return  initialize % locals()
      
    @staticmethod
    def body(compileResult, resultsym):
      code, decls = compileResult
      querydef_init_filled = querydef_init % locals()
      return emitlist(decls)+querydef_init_filled+code

    @staticmethod
    def finalize(resultsym):
        return  finalize % locals()

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

    @classmethod
    def compile_stringliteral(cls, s):
        #raise ValueError("String Literals not supported in C language: %s" % s)
        return """string_lookup("%s")""" % s

    @classmethod
    def negation(cls, input):
        return "(!%s)" % (input,)

    @classmethod
    def boolean_combine(cls, args, operator="&&"):
        opstr = " %s " % operator
        conjunc = opstr.join(["(%s)" % cls.compile_boolean(arg) for arg in args])
        LOG.debug("conjunc: %s", conjunc)
        return "( %s )" % conjunc

    @classmethod
    def compile_attribute(cls, expr):
        if isinstance(expr, expression.NamedAttributeRef):
            raise TypeError("Error compiling attribute reference %s. C compiler only support unnamed perspective.  Use helper function unnamed." % expr)
        if isinstance(expr, expression.UnnamedAttributeRef):
            symbol = expr.tupleref.name
            position = expr.position # NOTE: this will only work in Selects right now
            return '%s.get(%s)' % (symbol, position)

class CCOperator (Pipelined):
    language = CC
    
class MemoryScan(algebra.Scan, CCOperator):
  def produce(self):
    code = ""
    #generate the materialization from file into memory
    #TODO split the file scan apart from this in the physical plan

    #TODO for now this will break whatever relies on self.bound like reusescans
    #Scan is the only place where a relation is declared
    resultsym = gensym()
    
    code += FileScan(self.relation_key, self._scheme).compileme(resultsym)

    # now generate the scan from memory
    inputsym = resultsym

    #TODO: generate row variable to avoid naming conflict for nested scans
    memory_scan_template = """for (uint64_t i : %(inputsym)s->range()) {
          %(tuple_type)s %(tuple_name)s(%(inputsym)s, i);
          
          %(inner_plan_compiled)s
       } // end scan over %(inputsym)s
       """
    
    stagedTuple = StagedTupleRef(inputsym, self.scheme())

    tuple_type_def = stagedTuple.generateDefition()
    tuple_type = stagedTuple.getTupleTypename()
    tuple_name = stagedTuple.name
    
    inner_plan_compiled, inner_decls = self.parent.consume(stagedTuple, self)

    code += memory_scan_template % locals()
    return code, [tuple_type_def]+inner_decls
    
  def consume(self, t, src):
    assert False, "as a source, no need for consume"
    
    

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




class TwoPassSelect(algebra.Select, CCOperator):
    """
  Count matches, allocate memory, loop again to populate result
  """
  

    @classmethod
    def tagcondition(cls, condition, inputsym):
        """Tag each position reference in the join condition with the relation symbol it should refer to in the compiled code. joinlevel is the index of the join in the chain."""
        # TODO: this function is also impossible to understand.  Attribute references need an overhaul.
        def helper(condition):
            if isinstance(condition, expression.UnaryBooleanOperator):
                helper(condition.input)
            if isinstance(condition, expression.BinaryBooleanOperator):
                helper(condition.left)
                helper(condition.right)
            if isinstance(condition, expression.BinaryComparisonOperator):
                if isinstance(condition.left, expression.UnnamedAttributeRef):
                    condition.left.rowvariable = "i"
                    condition.left.relationsymbol = inputsym
                if isinstance(condition.right, expression.UnnamedAttributeRef):
                    condition.right.rowvariable = "i"
                    condition.right.relationsymbol = inputsym

        helper(condition)


    def compileme(self, resultsym, inputsym):
        LOG.debug("compiling %s of scheme %s", self.opname(), self.scheme())
        pcondition = CC.unnamed(self.condition, self.scheme())
        LOG.debug("CC.unnamed %s => %s", self.condition, pcondition)
        self.tagcondition(pcondition, inputsym)
        condition = CC.compile_boolean(pcondition)
        LOG.debug("CC.compile_boolean %s => %s", pcondition, condition)
        # Preston's original
        #code = twopass_select_template % locals()

        twopass_select_template = file(os.path.join(template_path,"select_simple_twopass.template")).read()
        code = twopass_select_template % locals()

        return code
    
    def __str__(self):
        return "%s[%s]" % (self.opname(), self.condition)

class HashJoin(algebra.Join, CCOperator):
  _i = 0

  @classmethod
  def __genHashName__(cls):
    name = "hash_%03d" % cls._i;
    cls._i += 1
    return name
  
  def produce(self):
    if not isinstance(self.condition, expression.EQ):
      msg = "The C compiler can only handle equi-join conditions of a single attribute: %s" % self.condition
      raise ValueError(msg)
    
    self._hashname = self.__genHashName__()
    self.outTuple = StagedTupleRef(gensym(), self.scheme())
    
    self.right.childtag = "right"
    code_right, decls_right = self.right.produce()
    
    self.left.childtag = "left"
    code_left, decls_left = self.left.produce()

    return code_right+code_left, decls_right+decls_left
  
  def consume(self, t, src):
    if src.childtag == "right":
      declr_template =  """std::unordered_map<int64_t, std::vector<%(in_tuple_type)s>* > %(hashname)s;
      """
      
      right_template = """insert(%(hashname)s, %(keyname)s, %(keypos)s);
      """   
      
      hashname = self._hashname
      keyname = t.name
      keypos = self.condition.right.position-len(self.left.scheme())
      
      out_tuple_type_def = self.outTuple.generateDefition()
      self.rightTuple = t #TODO: this induces a right->left dependency
      in_tuple_type = self.rightTuple.getTupleTypename()

      # declaration of hash map
      hashdeclr =  declr_template % locals()
      
      # materialization point
      code = right_template % locals()
      
      return code, [out_tuple_type_def,hashdeclr]
    
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
      #right_tuple_name = StagedTupleRef.genname() 

      out_tuple_type = self.outTuple.getTupleTypename()
      out_tuple_name =self.outTuple.name
      
      inner_plan_compiled, inner_plan_declrs = self.parent.consume(self.outTuple, self)
      
      code = left_template % locals()
      return code, inner_plan_declrs

    assert False, "src not equal to left or right"
      

class TwoPassHashJoin(algebra.Join, CCOperator):
    """
  A Join that hashes its left input and constructs an output relation.
  """
    def compileme(self, resultsym, leftsym, rightsym):
        if not isinstance(self.condition, expression.EQ):
            msg = "The C compiler can only handle equi-join conditions of a single attribute: %s" % self.condition
            raise ValueError(msg)

        template = "%s->relation"
        condition = CC.compile_boolean(self.condition)
        leftattribute = self.condition.left.position
        rightattribute = self.condition.right.position

        #leftattribute, rightattribute = self.attributes[0]
        #leftpos = self.scheme().getPosition(leftattribute)
        #leftattribute = CC.compile_attribute(leftpos, template % leftsym)
        #rightpos = self.scheme().getPosition(rightattribute)
        #rightattribute = CC.compile_attribute(rightpos, template % rightsym)

        code = hashjoin_template % locals()
        return code
      
    def __str__(self):
        return "%s(%s,%s,%s)[%s, %s]" % (self.opname(),
                                         self.condition,
                                         self.leftcondition,
                                         self.rightcondition,
                                         self.left,
                                         self.right)

class FilteringJoin(algebra.Join, CCOperator):
    """Abstract class representing a join that applies selection
  conditions on one or both inputs"""

    def __init__(self, condition=None, left=None, right=None, leftcondition=None, rightcondition=None):
        self.condition = condition
        self.leftcondition = leftcondition
        self.rightcondition = rightcondition
        algebra.BinaryOperator.__init__(self, left, right)

    def __str__(self):
        return "%s(%s,%s,%s)[%s, %s]" % (self.opname()
           , self.condition
           , self.leftcondition
           , self.rightcondition
           , self.left
           , self.right)

    def copy(self, other):
        self.condition = other.condition
        self.leftcondition = other.leftcondition
        self.rightcondition = other.rightcondition
        algebra.BinaryOperator.copy(self, other)


def compile_filtering_join(joincondition, leftcondition, rightcondition):
    """return code for processing a filtering join"""

class FilteringNestedLoopJoin(FilteringJoin, CCOperator):
    """
  A nested loop join that applies a selection condition on each relation.
  """
    def compileme(self, resultsym, leftsym, rightsym):
        if not isinstance(self.condition, expression.EQ):
            msg = "The C compiler can only handle equi-join conditions of a single attribute: %s" % self.condition
            raise ValueError(msg)

        template = "%s->relation"
        condition = CC.compile_boolean(self.condition)
        leftattribute = self.condition.left.position
        rightattribute = self.condition.right.position

        left_root_condition = self.leftcondition
        right_condition = self.rightcondition

        depth = 1
        inner_plan = leftsym

        relation_decls = """
    struct relationInfo *rel1 = %s;
    struct relationInfo *rel2 = %s;
    """ % (leftsym, rightsym)

        join_decls = """
    struct relationInfo *join%s_left = rel1;
    uint64 join1_leftattribute = %s;

    struct relationInfo *join%s_right = rel2;
    uint64 join1_rightattribute = %s;
    """ % (depth, leftattribute, depth, rightattribute)

        inner_plan_compiled = """


    printf("joined tuple: %d, %d\\n", join1_leftrow, join1_rightrow);
    resultcount++;


    """

        code = filtering_nestedloop_join_chain_template % locals()
        return code

def indentby(code, level):
    indent = " " * ((level + 1) * 6)
    return "\n".join([indent + line for line in code.split("\n")])

nextjointemplate = """
{ /* Begin Join Level %(depth)s */

  #pragma mta trace "running join %(depth)s"

  if (%(left_condition)s) { // filter on join%(depth)s.left
    // Join %(depth)s
    for (uint64 %(rightsym)s_row = 0; %(rightsym)s_row < %(rightsym)s->tuples; %(rightsym)s_row++) {
      if (%(right_condition)s) { // filter on join%(depth)s.right
        if (equals(%(leftsym)s, %(leftsym)s_row, %(leftposition)s // left attribute ref
                 , %(rightsym)s, %(rightsym)s_row, %(rightposition)s)) { //right attribtue ref

           %(inner_plan_compiled)s

        } // Join %(depth)s condition
      } // filter on join%(depth)s.right
    } // loop over join%(depth)s.right
  } // filter on join%(depth)s.left

}
"""

firstjointemplate = """
{ /* Begin Join Chain */

  printf("Begin Join Chain %(argsyms)s\\n");
  #pragma mta trace "running join %(depth)s"

  double start = timer();

  getCounters(counters, currCounter);
  currCounter = currCounter + 1; // 1

  // Loop over left leaf relation
  for (uint64 %(leftsym)s_row = 0; %(leftsym)s_row < %(leftsym)s->tuples; %(leftsym)s_row++) {

%(inner_plan_compiled)s

  } // loop over join%(depth)s.left

} // End Filtering_NestedLoop_Join_Chain
"""



class FilteringNLJoinChain(algebra.NaryJoin, CCOperator):
    """
  A linear chain of joins, with selection predicates applied"""
    def __init__(self, inputs, leftconditions, rightconditions, joinconditions):
        assert(len(rightconditions) == len(joinconditions))
        assert(len(rightconditions) == len(leftconditions))
        self.joinconditions = joinconditions
        self.leftconditions = leftconditions
        self.rightconditions = rightconditions
        self.finalcondition = expression.TAUTOLOGY
        algebra.NaryOperator.__init__(self, inputs)

    def __colToRelationAndOffset(self, column, argsyms):
        """from a global (among argsyms) column id, return the relation symbol and local column id.
        For example, if we join R(x,y),S(y,z), __colToRelationAndOffsetence(2) returns (S,0) and __colToRelationAndOffseterence(1) returns (R,1) (assuming S and R are relation symbols generated by the compiler"""
        position = column
        for i, arg in enumerate(self.args):
            offset = position - len(arg.scheme())
            if offset < 0:
                return (argsyms[i], i, position)
            position = offset

        raise IndexError("Column %s out of range of scheme %s" % (column, self.scheme()))

    @classmethod
    def rowvar(cls, relsym):
        return "%s_row" % relsym

    def tagcondition(self, joinlevel, condition, argsyms, conditiontype="join"):
        LOG.debug("tag condition %s,%s,%s,%s", joinlevel, condition, argsyms, self)
        """Tag each position reference in the join condition with the relation symbol it should refer to in the compiled code. joinlevel is the index of the join in the chain."""
        # TODO: this function is impossible to understand.  Attribute references need an overhaul.
        # TODO: May want to include a pipeline object that abstracts chains of non-blocking operators
        def helper(condition):
            if isinstance(condition, expression.UnaryBooleanOperator):
                helper(condition.input)
            if isinstance(condition, expression.BinaryBooleanOperator):
                helper(condition.left)
                helper(condition.right)
            if isinstance(condition, expression.BinaryComparisonOperator):

                def localize(posref):
                    assert(isinstance(posref, expression.UnnamedAttributeRef))
                    relsym, foundlevel, localposition = self.__colToRelationAndOffset(posref.position, argsyms)
                    rowvar = "%s_row" % relsym
                    return relsym, rowvar

                if conditiontype=="final":
                    if isinstance(condition.left, expression.NamedAttributeRef):
                        # FIXME: the assertion on localize() will always fail here
                        condition.left.relationsymbol, condition.left.rowvariable = localize(condition.left)
                    if isinstance(condition.right, expression.NamedAttributeRef):
                        condition.right.relationsymbol, condition.right.rowvariable = localize(condition.right)

                elif conditiontype=="join":
                    condition.left.relationsymbol, condition.left.rowvariable = localize(condition.left)
                    assert(isinstance(condition.right, expression.UnnamedAttributeRef))
                    relsym = argsyms[joinlevel + 1]
                    condition.right.relationsymbol = relsym
                    condition.right.rowvariable = self.rowvar(relsym)

                else: # right-hand selection
                    # selection condition
                    if conditiontype == "left":
                        relsym = argsyms[joinlevel]
                    else:
                        relsym = argsyms[joinlevel + 1]
                    if isinstance(condition.left, expression.NamedAttributeRef):
                        condition.left.relationsymbol = relsym
                        condition.left.rowvariable = self.rowvar(relsym)
                    if isinstance(condition.right, expression.NamedAttributeRef):
                        condition.right.relationsymbol = relsym
                        condition.right.rowvariable = self.rowvar(relsym)


        helper(condition)

    def compileme(self, resultsym, argsyms):
        LOG.debug("compiling %s: %s %s %s of scheme %s", self.__class__.__name__, resultsym, argsyms, self, self.scheme())
        def helper(level):
            depth = level
            if level < len(self.joinconditions):

                joincondition = self.joinconditions[level]
                if not isinstance(joincondition, expression.EQ):
                    raise ValueError("The C compiler can only handle equi-join conditions of a single attribute")

                assert(isinstance(joincondition.left, expression.UnnamedAttributeRef))
                assert(isinstance(joincondition.right, expression.UnnamedAttributeRef))

                # change the addressing scheme for the left-hand attribute reference
                LOG.debug("before tag join %s %s %s", joincondition, joincondition.left, joincondition.right)
                self.tagcondition(level, joincondition, argsyms, conditiontype="join")
                LOG.debug("after tag join %s %s %s", joincondition, joincondition.left.relationsymbol, joincondition.right.relationsymbol)
                leftsym = joincondition.left.relationsymbol
                leftposition = joincondition.left.position
                rightsym = joincondition.right.relationsymbol
                rightposition = joincondition.right.position

                left_condition = self.leftconditions[level]
                self.tagcondition(level, left_condition, argsyms, conditiontype="left")
                left_condition = CC.compile_boolean(left_condition)

                right_condition = self.rightconditions[level]
                self.tagcondition(level, right_condition, argsyms, conditiontype="right")
                right_condition = CC.compile_boolean(right_condition)

                inner_plan_compiled = helper(level+1)

                code = nextjointemplate % locals()

            else:
                depth = depth - 1
                if hasattr(self, "finalcondition"):
                    # TODO: Attribute references once again brittle and ugly
                    self.tagcondition(depth, self.finalcondition, argsyms, conditiontype="final")
                    condition = CC.compile_boolean(self.finalcondition)
                    wrapper = """
          if (%s) {%s}""" % (condition, "%s")
                else:
                    wrapper = "%s"
                N = len(self.rightconditions)
                ds = ", ".join(["%d" for d in range(N)])
                rowvars = ", ".join(["%s_row" % d for d in argsyms])
                code = wrapper % """

          // Here we would send the tuple to the client, or write to disk, or fill a data structure
          printf("joined tuple: %%d, %(ds)s\\n", %(rowvars)s);
          resultcount++;

        """ % locals()
            return indentby(code, depth)

        depth = 0
        # get left relation
        leftsym = argsyms[depth]
        # get attribute position in left relation
        firstjoin = self.joinconditions[depth]
        assert(isinstance(firstjoin.left, expression.UnnamedAttributeRef))
        leftposition = self.joinconditions[depth].left.position
        # get condition on left relation
        left_condition = self.leftconditions[depth]
        self.tagcondition(depth, left_condition, argsyms, conditiontype="left")
        left_condition = CC.compile_boolean(left_condition)

        inner_plan_compiled = helper(depth)
        code = firstjointemplate % locals()
        return code

    def __str__(self):
        args = ",".join(["%s" % arg for arg in self.args])
        # TODO: clean up this final condition nonsense
        return "FilteringNLJoinChain(%s, %s, %s, %s)[%s]" % (self.joinconditions, self.leftconditions, self.rightconditions, self.finalcondition, args)
      
    def shortStr(self):
      return "FilteringNLJoinChain[%s]" % self.args


# iteration  over table + insertion into hash table with filter
hash_build_template = """
// build hash table for %(hashedsym)s, column %(position)s
hash_table_t<uint64_t, uint64_t> %(hashedsym)s_hash_%(position)s;
for (uint64_t  %(hashedsym)s_row = 0; %(hashedsym)s_row < %(hashedsym)s->tuples; %(hashedsym)s_row++) {
  if (%(condition)s) {
    insert(%(hashedsym)s_hash_%(position)s, %(hashedsym)s, %(hashedsym)s_row, %(position)s);
  }
}
"""

# iteration over table + lookup
# name of tuple output is r{depth}
# note the the rightcondition, is the condition in the build phase
nested_hash_join_first_template = """
for (uint64_t  %(leftsym)s_row = 0; %(leftsym)s_row < %(leftsym)s->tuples; %(leftsym)s_row++) {
  if (%(leftcondition)s) {
    for (Tuple<uint64_t> r%(depth)s : lookup(%(rightsym)s_hash_%(rightposition)s, %(leftsym)s, %(leftposition)s)) {
      %(inner_plan_compiled)s
    } // end join of %(leftsym)s[%(leftposition)s] and %(rightsym)s[%(rightposition)s]
  } // end filter %(depth)s
} // end scan over %(leftsym)s
"""

# lookup tuple
nested_hash_join_rest_template = """
if (%(leftcondition)s) {
  for (Tuple<uint64_t> r%(depth)s : lookup(%(rightsym)s_hash_%(rightposition)s, r%(depthMinusOne)s, %(leftposition)s) {
    %(inner_plan_compiled)s
  }
}
"""
class FilteringHashJoinChain(algebra.NaryJoin, CCOperator):
    """
  A linear chain of joins, with selection predicates applied"""
    def __init__(self, inputs, leftconditions, rightconditions, joinconditions):
        assert(len(rightconditions) == len(joinconditions))
        assert(len(rightconditions) == len(leftconditions))
        self.joinconditions = joinconditions
        self.leftconditions = leftconditions
        self.rightconditions = rightconditions
        self.finalcondition = expression.TAUTOLOGY
        algebra.NaryOperator.__init__(self, inputs)
        
    def shortStr(self):
      return "%s(%s,%s,%s)[...]" % (self.opname(), 
                                   self.joinconditions, 
                                   self.leftconditions, 
                                   self.rightconditions)

    def __str__(self):
      return "%s(%s,%s,%s)[%s]" % (self.opname(), 
                                   self.joinconditions, 
                                   self.leftconditions, 
                                   self.rightconditions,
                                   self.args)

    def __colToRelationAndOffset(self, column, argsyms):
        """from a global column id, return the relation symbol and local column id.
        For example, if we join R(x,y),S(y,z), __colToRelationAndOffsetence(2) returns (S,0) and __colToRelationAndOffseterence(1) returns (R,1) (assuming S and R are relation symbols generated by the compiler"""
        position = column
        for i, arg in enumerate(self.args):
            offset = position - len(arg.scheme())
            if offset < 0:
                return (argsyms[i], i, position)
            position = offset

        raise IndexError("Column %s out of range of scheme %s" % (column, self.scheme()))
        
    def compileme(self, resultsym, argsyms):
      code = ""
      
      # find the right columns that need to be hashed
      hash_columns = [c.right for c in self.joinconditions]
      LOG.debug("hash columns %s", hash_columns)
     
      # generate hash builds 
      for i, col in enumerate(hash_columns):
        hashedsym, position, _ = self.__colToRelationAndOffset(col.position, argsyms)
        condition = CC.compile_boolean(self.rightconditions[i])
        LOG.debug(condition)
        LOG.debug(hash_build_template)
        LOG.debug(locals())
        code += hash_build_template % locals()
      
      
      def helper(depth):
        if depth == len(self.joinconditions):
          return indentby(self.language.comment("TODO: emit the result"), depth)

        rightsym, rightposition, _ = self.__colToRelationAndOffset(self.joinconditions[depth].right.position, argsyms)
        leftsym, leftposition, _ = self.__colToRelationAndOffset(self.joinconditions[depth].left.position, argsyms)
        leftcondition = CC.compile_boolean(self.leftconditions[depth])

        if depth == 0:
          inner_plan_compiled = helper(depth+1)
          newcode = nested_hash_join_first_template % locals()
          return newcode
        else:
          inner_plan_compiled = helper(depth+1)
          depthMinusOne = depth-1
          newcode = nested_hash_join_rest_template % locals()
          return indentby(newcode, depth)
          
      code += helper(0) 
      return code
    


class FilteringHashJoin(FilteringJoin, CCOperator):
    """
  A Join that applies a selection condition to each of its inputs.
  UNFINISHED
  TODO: Remove the use of attributes and update to boolean condition
  """
    def compileme(self, resultsym, leftsym, rightsym):
        if len(self.attributes) > 1: raise ValueError("The C compiler can only handle equi-join conditions of a single attribute")

        pcondition = CC.unnamed(self.condition, self.scheme())
        condition = CC.compile_boolean(pcondition)

        template = "%s->relation"
        leftattribute, rightattribute = self.attributes[0]
        leftpos = self.scheme().getPosition(leftattribute)
        leftattribute = CC.compile_attribute(leftpos, template % leftsym)
        rightpos = self.scheme().getPosition(rightattribute)
        rightattribute = CC.compile_attribute(rightpos, template % rightsym)

        code = filteringhashjoin_template % locals()
        return code

#def neededDownstream(ops, expr):
#  if expr in ops:
#
#
#
#class FreeMemory(CCOperator):
#  def fire(self, expr):
#    for ref in noReferences(expr)

class FilteringHashJoinChainRule(rules.Rule):
  def fire(self, expr):
    if isinstance(expr, FilteringNLJoinChain):
      assert False, "TODO: Need to check condition for UnnamedAttr==UnnamedAttr"
      return FilteringHashJoinChain(expr.args,
                                    expr.leftconditions,
                                    expr.rightconditions, 
                                    expr.joinconditions)
    else:
      return expr
    
  def __str__(self):
    return "FilteringNLJoinChain => FilteringHashJoinChain"

class FilteringNestedLoopJoinRule(rules.Rule):
    """A rewrite rule for combining Select and Join"""
    def fire(self, expr):

        if isinstance(expr, algebra.Join):
            left = isinstance(expr.left, algebra.Select)
            right = isinstance(expr.right, algebra.Select)
            taut = expression.TAUTOLOGY
            if left and right:
                return FilteringNestedLoopJoin(expr.condition
                                        ,expr.left.input
                                        ,expr.right.input
                                        ,expr.left.condition
                                        ,expr.right.condition)
            if left:
                return FilteringNestedLoopJoin(expr.condition
                                        ,expr.left.input
                                        ,expr.right
                                        ,expr.left.condition
                                        ,taut)
            if right:
                return FilteringNestedLoopJoin(expr.condition
                                        ,expr.left
                                        ,expr.right.input
                                        ,taut
                                        ,expr.right.condition)
            else:
                return FilteringNestedLoopJoin(expr.condition
                                        ,expr.left
                                        ,expr.right
                                        ,taut
                                        ,taut)

        return expr

    def __str__(self):
        return "Join(Select, Select) => FilteringJoin"

class LeftDeepFilteringJoinChainRule(rules.Rule):
    """A rewrite rule for combining Select(Join(Select, Select)*) into one pipeline."""
    """Turns separate FilteringNestedLoopJoins into a single FilteringNLJoinChain"""
    def fire(self, expr):
        topoperator = expr
        if isinstance(expr, algebra.Select):
            finalselect = topoperator
            topoperator = expr.input
        else:
            finalselect = None
        def helper(expr, joinchain):
            """Follow a left deep chain of joins, gathering conditions"""
            if isinstance(expr, FilteringNestedLoopJoin):
                # push args and conditions onto the front
                joinchain.args[0:0] = [expr.right]
                LOG.debug("joinchain.args = %s", joinchain.args)
                joinchain.joinconditions[0:0] = [expr.condition]
                joinchain.leftconditions[0:0] = [expr.leftcondition]
                joinchain.rightconditions[0:0] = [expr.rightcondition]
                return helper(expr.left, joinchain)
            else:
                if finalselect:
                    joinchain.finalcondition = finalselect.condition
                joinchain.args[0:0] = [expr]
                return joinchain

        if isinstance(topoperator, FilteringNestedLoopJoin):
            joinchain = FilteringNLJoinChain([], [], [], [])
            LOG.debug("before  select+join %s %s", topoperator, joinchain)
            newexpr = helper(topoperator, joinchain)
            LOG.debug("after selct+join %s", joinchain)
            return newexpr
        else:
            return expr
      
  
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
