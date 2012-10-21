import boolean
import rules
import algebra
from language import Language
import catalog

import os.path

template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_templates")

def readtemplate(fname):
  return file(os.path.join(template_path,fname)).read()

template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_templates")

base_template = readtemplate("base_query.template")
initialize,finalize = base_template.split("// SPLIT ME HERE")
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
""" % (cls.relation_decl(rvar), cls.assignment(rvar,val))

  @classmethod
  def relation_decl(cls, rvar):
    return "struct relationInfo *%s;"

  @classmethod
  def assignment(cls, x, y):
    return "%s = %s;" % (x,y)

  @staticmethod
  def initialize(resultsym):
    return  initialize % locals()

  @staticmethod
  def finalize(resultsym):
    return  finalize % locals()

  @classmethod
  def compile_stringliteral(cls, s):
    raise ValueError("String Literals not supported in C language: %s" % s)
    #return """string_lookup("%s")""" % s

  @classmethod
  def boolean_combine(cls, args, operator="&&"):
    opstr = " %s " % operator 
    conjunc = opstr.join(["(%s)" % cls.compile_boolean(arg) for arg in args])
    return "( %s )" % conjunc

  @classmethod
  def compile_attribute(cls, position, relation="relation"):
    return '%s[i + %s]' % (relation, position)

class CCOperator:
  language = CC

class FileScan(algebra.Scan, CCOperator):
  def compileme(self, resultsym):
    name = self.relation.name
    if isinstance(self.relation, catalog.ASCIIFile):
      code = ascii_scan_template % locals()
    else:
      code = binary_scan_template % locals()
    return code

class TwoPassSelect(algebra.Select, CCOperator):
  """
Count matches, allocate memory, loop again to populate result
"""
  def compileme(self, resultsym, inputsym):
    pcondition = CC.unnamed(self.condition, self.scheme())
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
    if not isinstance(self.condition,boolean.EQ):
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


class FilteringNestedLoopJoin(FilteringJoin, CCOperator):
  """
A nested loop join that applies a selection condition on each relation.
"""
  def compileme(self, resultsym, leftsym, rightsym):
    if not isinstance(self.condition,boolean.EQ):
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

class FilteringHashJoin(FilteringJoin, CCOperator):
  """
A Join that applies a selection condition to each of its inputs.
UNFINISHED
TODO: Remove the use of attributes and update to boolean condition
"""
  def compileme(self, resultsym, leftsym, rightsym):
    if len(self.butes) > 1: raise ValueError("The C compiler can only handle equi-join conditions of a single attribute")

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
                                ,boolean.EQ(1,1))
      if right:
        return FilteringNestedLoopJoin(expr.condition
                                ,expr.left
                                ,expr.right.input
                                ,boolean.EQ(1,1)
                                ,expr.right.condition)
      else:
        return FilteringNestedLoopJoin(expr.condition
                                ,expr.left
                                ,expr.right
                                ,boolean.EQ(1,1)
                                ,boolean.EQ(1,1))

    return expr

  def __str__(self):
    return "Project => ()"

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
  FilteringNestedLoopJoinRule(),
  rules.OneToOne(algebra.Select,TwoPassSelect),
  rules.OneToOne(algebra.Scan,FileScan),
#  rules.FreeMemory()
]
