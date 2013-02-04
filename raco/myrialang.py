import boolean
import rules
import algebra
import json
from language import Language

class MyriaLanguage(Language):
  @classmethod
  def new_relation_assignment(cls, rvar, val):
    return """
%s
%s
""" % (cls.relation_decl(rvar), cls.assignment(rvar,val))

  @classmethod
  def relation_decl(cls, rvar):
    # no type declarations necessary
    return ""

  @staticmethod
  def assignment(x, y):
    return ""

  @staticmethod
  def comment(txt):
    #return  "# %s" % txt
    # comments not technically allowed in json
    return  ""

  @staticmethod
  def preamble(query=None, plan=None):
    return  """
{
  "raw_datalog" : "%s",
  "logical_ra" : "%s",
  "query_plan" : {
""" % (query, plan)

  @staticmethod
  def initialize(resultsym):
    return  """
    "%s" : [[
""" % resultsym

  @staticmethod
  def finalize(resultsym):
    return  """
      ]]
"""

  @staticmethod
  def postamble(query=None, plan=None):
    return  """
  }
}
""" 

  @classmethod
  def boolean_combine(cls, args, operator="and"):
    opstr = " %s " % operator 
    conjunc = opstr.join(["%s" % cls.compile_boolean(arg) for arg in args])
    return "(%s)" % conjunc

  @staticmethod
  def mklambda(body, var="t"):
    return ("lambda %s: " % var) + body

  @staticmethod
  def compile_attribute(name):
    return '%s' % name

class MyriaOperator:
  language = MyriaLanguage

class MyriaScan(algebra.Scan, MyriaOperator):
  def compileme(self, resultsym):
    return """
{ 
  "op_name" : "%s",
  "op_type" : "SCAN", 
  "arg_relation_name" : "%s"
},""" % (resultsym, self.relation.name)

class MyriaSelect(algebra.Select, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    """
{ 
  "name" : "%s",
  "type" : "SELECT",
  "condition" : "%s",
  "input" : "%s"
},""" % (resultsym, self.language.compile_boolean(self.condition), inputsym)

class MyriaProject(algebra.Project, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    cols = [str(x) for x in self.columnlist]
    return """
{ 
  "op_name" : "%s",
  "op_type" : "PROJECT",
  "columnlist" : %s,
  "arg_child" : "%s"
},""" % (resultsym, cols, inputsym)

class MyriaInsert(algebra.Store, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    return """
{ 
  "op_name" : "%s",
  "op_type" : "INSERT",
  "arg_child" : "%s"
},""" % (resultsym, inputsym)


class MyriaEquiJoin(algebra.Join, MyriaOperator):

  @classmethod
  def convertcondition(self, condition):
    """Convert the joincondition to a list of left columns and a list of right columns representing a conjunction"""

    if isinstance(condition, boolean.AND):
      leftcols1, rightcols1 = self.convertcondition(condition.left)
      leftcols2, rightcols2 = self.convertcondition(condition.right)
      return leftcols1 + leftcols2, rightcols1 + rightcols2

    if isinstance(condition, boolean.EQ):
      return [str(condition.left.position)], [str(condition.right.position)]
  
  def compileme(self, resultsym, leftsym, rightsym):
    """Compile the operator to a sequence of json operators"""
  
    leftcols, rightcols = self.convertcondition(self.condition)

    def mkshuffle(symbol, joincond):
      return """
{ 
  "op_name" : "%s_scatter",
  "op_type" : "SHUFFLE_PRODUCER",
  "partition" : %s,
  "arg_child" : %s
}""" % (symbol, joincond, symbol)

    shuffleleft = mkshuffle(leftsym, leftcols)
    shuffleright = mkshuffle(rightsym, rightcols)
    shuffleconsume = """
{
  "op_name" : "%s_gather",
  "op_type" : "SHUFFLE_CONSUMER",
  "producers" : ["%s_scatter","%s_scatter"]
}""" % (resultsym, leftsym, rightsym)

    join = """
{
  "op_name" : "%s",
  "op_type" : "LocalJoin",
  "arg_child1" : "%s",
  "arg_columns1" : %s,
  "arg_child2": "%s",
  "arg_columns2" : %s,
}""" % (resultsym, leftsym, leftcols, rightsym, rightcols)
    return ",\n".join([shuffleleft, shuffleright, shuffleconsume, join])

class MyriaShuffle(algebra.PartitionBy,MyriaOperator):
  def compileme(self, resultsym, inputsym):
    shuffle = """
{"name" : "%s",
 "type" : "SHUFFLEPRODUCER",
 "hash" : "%s",
 "replicate" : "%s",
 "input": "%s"
}""" % (resultsym, self.columnlist, "not used", inputsym)
    return shuffle

class MyriaAlgebra:
  language = MyriaLanguage

  operators = [
  MyriaShuffle,
  MyriaEquiJoin,
  MyriaSelect,
  MyriaProject,
  MyriaScan
]
  rules = [
  rules.OneToOne(algebra.PartitionBy,MyriaShuffle),
  rules.OneToOne(algebra.Broadcast,MyriaShuffle),
  rules.OneToOne(algebra.Store,MyriaInsert),
  rules.OneToOne(algebra.Join,MyriaEquiJoin),
  rules.OneToOne(algebra.Select,MyriaSelect),
  rules.OneToOne(algebra.Project,MyriaProject),
  rules.OneToOne(algebra.Scan,MyriaScan)
]
 
