# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
#
# The above modeline for Vim automatically sets it to
# .. Bill's Python style. If it doesn't work, check
#     :set modeline?        -> should be true
#     :set modelines?       -> should be > 0

import boolean
import rules
import algebra
import json
from language import Language

def json_pretty_print(dictionary):
    """a function to pretty-print a JSON dictionary.
From http://docs.python.org/2/library/json.html"""
    return json.dumps(dictionary, sort_keys=True, 
            indent=2, separators=(',', ': '))

class MyriaLanguage(Language):
  # TODO: get the workers from somewhere
  workers = [1,2]
  reusescans = False

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
  "expected_result_size" : "-1",
  "query_plan" : {
""" % (query, plan)

  @staticmethod
  def initialize(resultsym):
    return ""
    return  """
    "%s" : [[
""" % resultsym

  @staticmethod
  def finalize(resultsym):
    return ""
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
  workers = MyriaLanguage.workers
  language = MyriaLanguage

class MyriaScan(algebra.Scan, MyriaOperator):
  def compileme(self, resultsym):
    return json_pretty_print({
        "op_name" : resultsym,
        "op_type" : "SCAN",
        "arg_user_name" : "public",
        "arg_program_name" : "adhoc",
        "arg_relation_name" : self.relation.name
      })+','

class MyriaSelect(algebra.Select, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    return json_pretty_print({
        "op_name" : resultsym,
        "op_type" : "SELECT",
        "condition" : self.language.compile_boolean(self.condition),
        "input" : inputsym
      }) + ','

class MyriaProject(algebra.Project, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    cols = [str(x) for x in self.columnlist]
    return json_pretty_print({
        "op_name" : resultsym,
        "op_type" : "PROJECT",
        "columnlist" : cols,
        "arg_child" : inputsym
      })+','

class MyriaInsert(algebra.Store, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    return json_pretty_print({
        "op_name" : resultsym,
        "op_type" : "INSERT",
        "arg_child" : inputsym
      })+','

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

    def mkshuffle(symbol, op_id, joincond):
      return json_pretty_print({
          "op_name" : "%s_scatter" % (symbol,),
          "op_type" : "ShuffleProducer",
          "arg_child" : symbol,
          "arg_workerIDs" : self.workers,
          "arg_operatorID" : op_id,
          "arg_pf" : ["SingleFieldHash", joincond]
        })+','

    shuffleleft = mkshuffle(leftsym, 0, leftcols)
    shuffleright = mkshuffle(rightsym, 1, rightcols)

    def mkconsumer(symbol,scheme,op_id):
      return json_pretty_print({
          "op_name" : "%s_gather" % (symbol,),
          "op_type" : "ShuffleConsumer",
          "arg_schema" : scheme,
          "arg_workerIDs" : self.workers,
          "arg_operatorID" : op_id
        })+','

    def pretty(s):
      names, descrs = zip(*s.asdict.items())
      names = ["%s" % n for n in names]
      types = [r[1] for r in descrs]
      return {"column_types" : types, "column_names" : names}

    consumeleft = mkconsumer(leftsym, pretty(self.left.scheme()),  0)
    consumeright = mkconsumer(rightsym, pretty(self.right.scheme()), 1)

    cols = [str(i) for i in range(len(self.scheme()))]
    allleft = cols[:len(self.left.scheme())]
    allright = cols[len(self.right.scheme()):]

    join = json_pretty_print({
        "op_name" : resultsym,
        "op_type" : "LocalJoin",
        "arg_child1" : "%s_gather" % (leftsym,),
        "arg_columns1" : leftcols,
        "arg_child2": "%s_gather" % (rightsym,),
        "arg_columns2" : rightcols,
        "arg_select1" : allleft,
        "arg_select2" : allright
      })+','

    return "\n".join([shuffleleft, shuffleright, consumeleft, consumeright, join])

class MyriaShuffle(algebra.PartitionBy,MyriaOperator):
  def compileme(self, resultsym, inputsym):
    scatter = json_pretty_print({
        "op_name" : "%s_scatter" % (resultsym,),
        "op_type" : "SHUFFLE_PRODUCER",
        "partition" : self.columnlist,
        "arg_child": inputsym
      })+','
    gather = json_pretty_print({
        "op_name" : "%s_gather" % (resultsym,),
        "op_type" : "SHUFFLE_CONSUMER",
        "producers" : ["%s_producer" % (resultsym,)]
      })+','
    return scatter + '\n' + gather

class MyriaParallel(algebra.ZeroaryOperator, MyriaOperator):
  """Turns a single plan into a forst of identical plans, one for each worker."""
  """An awkward operator.  It compiles itself instead of letting the compiler do it."""
  def __init__(self, plans):
    self.plans = plans
    algebra.Operator.__init__(self)

  def __repr__(self):
      return 'MyriaParallel(%s)' % repr(self.plans)

  def compileme(self, resultsym):
    patt = """
"%s" : [[
 %s 
]]
""" 
    def compile(p):
      algebra.reset()
      return p.compile(resultsym)
    return ",\n".join([patt % (worker, compile(plan)) for (worker, plan) in self.plans])

class BroadcastRule(rules.Rule):
  """Convert a broadcast operator to a shuffle"""
  def fire(self, expr):
    if isinstance(expr, algebra.Broadcast):
      columnlist = [boolean.PositionReference(i) for i in range(len(expr.scheme()))]
      newop = MyriaShuffle(columnlist, expr.input)
      return newop
    return expr

  def __str__(self):
    return "Project => ()"

class Parallel(rules.Rule):
  """Repeat a plan for each worker"""
  def __init__(self,N):
    self.workers = range(N)
 
  def fire(self, expr):
    def copy(expr):
      newop = expr.__class__()
      newop.copy(expr)
      return newop
    return MyriaParallel([(i, copy(expr)) for i in self.workers])
      

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
  BroadcastRule(),
  rules.OneToOne(algebra.Store,MyriaInsert),
  rules.OneToOne(algebra.Join,MyriaEquiJoin),
  rules.OneToOne(algebra.Select,MyriaSelect),
  rules.OneToOne(algebra.Project,MyriaProject),
  rules.OneToOne(algebra.Scan,MyriaScan),
  Parallel(2)
]
 
