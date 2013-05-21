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
from utility import emit

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
    return emit(cls.relation_decl(rvar), cls.assignment(rvar,val))

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
    return {
        "op_name" : resultsym,
        "op_type" : "SCAN",
        "arg_relation_key" : {
          "user_name" : "public",
          "program_name" : "adhoc",
          "relation_name" : self.relation.name
        }
      }

class MyriaSelect(algebra.Select, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    return {
        "op_name" : resultsym,
        "op_type" : "SELECT",
        "condition" : self.language.compile_boolean(self.condition),
        "input" : inputsym
      }

class MyriaProject(algebra.Project, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    cols = [x.position for x in self.columnlist]
    return {
        "op_name" : resultsym,
        "op_type" : "PROJECT",
        "columnlist" : cols,
        "arg_child" : inputsym
      }

class MyriaInsert(algebra.Store, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    return {
        "op_name" : resultsym,
        "op_type" : "INSERT",
        "arg_child" : inputsym
      }

class MyriaJoin(algebra.ProjectingJoin, MyriaOperator):

  @classmethod
  def convertcondition(self, condition):
    """Convert the joincondition to a list of left columns and a list of right columns representing a conjunction"""

    if isinstance(condition, boolean.AND):
      leftcols1, rightcols1 = self.convertcondition(condition.left)
      leftcols2, rightcols2 = self.convertcondition(condition.right)
      return leftcols1 + leftcols2, rightcols1 + rightcols2

    if isinstance(condition, boolean.EQ):
      return [condition.left.position], [condition.right.position]

    raise NotImplementedError("Myria only supports EquiJoins")
  
  def compileme(self, resultsym, leftsym, rightsym):
    """Compile the operator to a sequence of json operators"""
  
    leftcols, rightcols = self.convertcondition(self.condition)

    def mkshuffle(symbol, op_id, joincond):
      return {
          "op_name" : "%s_scatter" % symbol,
          "op_type" : "ShuffleProducer",
          "arg_child" : symbol,
          "arg_worker_ids" : self.workers,
          "arg_operator_id" : op_id,
          "arg_pf" : ["SingleFieldHash", joincond]
        }

    shuffleleft = mkshuffle(leftsym, 0, leftcols)
    shuffleright = mkshuffle(rightsym, 1, rightcols)

    def mkconsumer(symbol,scheme,op_id):
      return {
          "op_name" : "%s_gather" % (symbol,),
          "op_type" : "ShuffleConsumer",
          "arg_schema" : scheme,
          "arg_worker_ids" : self.workers,
          "arg_operator_id" : op_id
        }

    def pretty(s):
      names, descrs = zip(*s.asdict.items())
      names = ["%s" % n for n in names]
      types = [r[1] for r in descrs]
      return {"column_types" : types, "column_names" : names}

    consumeleft = mkconsumer(leftsym, pretty(self.left.scheme()),  0)
    consumeright = mkconsumer(rightsym, pretty(self.right.scheme()), 1)

    
    if self.columnlist is None:
      self.columnlist = [boolean.PositionReference(i) for i in xrange(len(self.scheme()))]
    allleft = [i.position for i in self.columnlist if i.position < len(self.left.scheme())]
    allright = [i.position-len(self.left.scheme()) for i in self.columnlist if i.position >= len(self.left.scheme())]

    join = {
        "op_name" : resultsym,
        "op_type" : "LocalJoin",
        "arg_child1" : "%s_gather" % (leftsym,),
        "arg_columns1" : leftcols,
        "arg_child2": "%s_gather" % (rightsym,),
        "arg_columns2" : rightcols,
        "arg_select1" : allleft,
        "arg_select2" : allright
      }

    return [shuffleleft, shuffleright, consumeleft, consumeright, join]

class MyriaParallel(algebra.ZeroaryOperator, MyriaOperator):
  """Turns a single plan into a forst of identical plans, one for each worker."""
  """An awkward operator.  It compiles itself instead of letting the compiler do it."""
  def __init__(self, plans):
    self.plans = plans
    algebra.Operator.__init__(self)

  def __repr__(self):
      return 'MyriaParallel(%s)' % repr(self.plans)

  def compileme(self, resultsym):
    def compile(p):
      algebra.reset()
      return p.compile(resultsym)
    ret = [{ worker : compile(plan) } for (worker, plan) in self.plans]
    return ret

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
      MyriaJoin
      , MyriaSelect
      , MyriaProject
      , MyriaScan
  ]

  rules = [
      rules.ProjectingJoin()
      , rules.OneToOne(algebra.Store,MyriaInsert)
      , rules.OneToOne(algebra.Join,MyriaJoin)
      , rules.OneToOne(algebra.Select,MyriaSelect)
      , rules.OneToOne(algebra.Project,MyriaProject)
      , rules.OneToOne(algebra.ProjectingJoin,MyriaJoin)
      , rules.OneToOne(algebra.Scan,MyriaScan)
      #, Parallel(2)
  ]
