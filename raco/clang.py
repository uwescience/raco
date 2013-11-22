import rules
import algebra
from language import Language
import catalog

import expression
import os.path

template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_templates")

def readtemplate(fname):
  return file(os.path.join(template_path, fname)).read()

template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_templates")

base_template = readtemplate("base_query.template")
initialize, finalize = base_template.split("// SPLIT ME HERE")
twopass_select_template = readtemplate("precount_select.template")
hashjoin_template = readtemplate("hashjoin.template")
filtering_nestedloop_join_chain_template = readtemplate("filtering_nestedloop_join_chain.template")
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
  def finalize(resultsym):
    return  finalize % locals()

  @staticmethod
  def log(txt):
    return  """printf("%s\\n");\n""" % txt

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
    return "( %s )" % conjunc

  @classmethod
  def compile_attribute(cls, expr):
    if isinstance(expr, expression.NamedAttributeRef):
      raise TypeError("Error compiling attribute reference %s. C compiler only support unnamed perspective.  Use helper function unnamed." % expr)
    if isinstance(expr, expression.UnnamedAttributeRef):
      position = expr.leaf_position
      relation = expr.relationsymbol
      rowvariable = expr.rowvariable
      return '%s->relation[%s*%s->fields + %s]' % (relation, rowvariable, relation, position)

class CCOperator (object):
  language = CC

class FileScan(algebra.Scan, CCOperator):

  def compileme(self, resultsym):
    name = self.relation_key
    #tup = (resultsym, self.originalterm.originalorder, self.originalterm)
    #self.trace("// Original query position of %s: term %s (%s)" % tup)

    if isinstance(self.relation_key, catalog.ASCIIFile):
      code = ascii_scan_template % locals()
    else:
      code = binary_scan_template % locals()
    return code

class TwoPassSelect(algebra.Select, CCOperator):
  """
Count matches, allocate memory, loop again to populate result
"""

  def tagcondition(self, condition, inputsym):
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
    pcondition = CC.unnamed(self.condition, self.scheme())
    self.tagcondition(pcondition, inputsym)
    condition = CC.compile_boolean(pcondition)
    # Preston's original
    #code = twopass_select_template % locals()

    twopass_select_template = file(os.path.join(template_path,"select_simple_twopass.template")).read()
    code = twopass_select_template % locals()

    return code


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

  def leafreference(self, column, argsyms):
    """return the relation symbol and local offset corresponding to column position "column."  For example, if we join R(x,y),S(y,z), leafof(2) returns (S,0) and leafof(1) returns (R,1) (assuming S and R are relation symbols generated by the compiler"""
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
          relsym, foundlevel, localposition = self.leafreference(posref.position, argsyms)
          rowvar = "%s_row" % relsym
          return relsym, rowvar

        if conditiontype=="final":
          if isinstance(condition.left, expression.NamedAttributeRef):
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
    def helper(level):
      depth = level
      if level < len(self.joinconditions):

        joincondition = self.joinconditions[level]
        if not isinstance(joincondition, expression.EQ):
          raise ValueError("The C compiler can only handle equi-join conditions of a single attribute")

        assert(isinstance(joincondition.left, expression.UnnamedAttributeRef))
        assert(isinstance(joincondition.right, expression.UnnamedAttributeRef))

        # change the addressing scheme for the left-hand attribute reference
        self.tagcondition(level, joincondition, argsyms, conditiontype="join")
        leftsym = joincondition.left.relationsymbol
        leftposition = joincondition.left.leaf_position
        rightsym = joincondition.right.relationsymbol
        rightposition = joincondition.right.leaf_position

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
    leftposition = self.joinconditions[depth].left.leaf_position
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
        joinchain.args[0:0] = [expr.right]
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
      return helper(topoperator, joinchain)
    else:
      return expr

class CCAlgebra:
  language = CC

  operators = [
  #TwoPassHashJoin,
  FilteringNestedLoopJoin,
  TwoPassSelect,
  FileScan
]
  rules = [
   #rules.OneToOne(algebra.Join,TwoPassHashJoin),
  rules.removeProject(),
  rules.CrossProduct2Join(),
  FilteringNestedLoopJoinRule(),
  LeftDeepFilteringJoinChainRule(),
  rules.OneToOne(algebra.Select,TwoPassSelect),
  rules.OneToOne(algebra.Scan,FileScan),
#  rules.FreeMemory()
]
