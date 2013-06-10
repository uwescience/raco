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

op_id = 0
def gen_op_id():
  global op_id
  op_id += 1
  return "operator%d" % op_id

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

class MyriaSQLiteScan(algebra.Scan, MyriaOperator):
  def compileme(self, resultsym):
    return {
        "op_name" : resultsym,
        "op_type" : "SQLiteScan",
        "relation_key" : {
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

class MyriaSQLiteInsert(algebra.Store, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    return {
        "op_name" : resultsym,
        "op_type" : "SQLiteInsert",
        "relation_key" : {
          "user_name" : "public",
          "program_name" : "adhoc",
          "relation_name" : self.relation.name
        },
        "arg_child" : inputsym
      }

class MyriaLocalJoin(algebra.ProjectingJoin, MyriaOperator):
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

    if self.columnlist is None:
      self.columnlist = [boolean.PositionReference(i) for i in xrange(len(self.scheme()))]

    allleft = [i.position for i in self.columnlist if i.position < len(self.left.scheme())]
    allright = [i.position-len(self.left.scheme()) for i in self.columnlist if i.position >= len(self.left.scheme())]

    join = {
        "op_name" : resultsym,
        "op_type" : "LocalJoin",
        "arg_child1" : "%s" % leftsym,
        "arg_columns1" : leftcols,
        "arg_child2": "%s" % rightsym,
        "arg_columns2" : rightcols,
        "arg_select1" : allleft,
        "arg_select2" : allright
      }

    return join

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

class MyriaShuffle(algebra.Shuffle, MyriaOperator):
  """Represents a simple shuffle operator"""
  def compileme(self, resultsym, inputsym):
    raise NotImplementedError('shouldn''t ever get here, should be turned into SP-SC pair')

class MyriaShuffleProducer(algebra.UnaryOperator, MyriaOperator):
  """A Myria ShuffleProducer"""
  def __init__(self, input, opid, hash_columns):
    algebra.UnaryOperator.__init__(self, input)
    self.opid = opid
    self.hash_columns = hash_columns

  def compileme(self, resultsym, inputsym):
    if len(self.hash_columns) == 1:
      pf = {
          "type" : "SingleFieldHash",
          "index" : self.hash_columns[0]
        }
    else:
      pf = {
          "type" : "MultiFieldHash",
          "index" : self.hash_columns
        }

    return {
        "op_name" : resultsym,
        "op_type" : "ShuffleProducer",
        "arg_child" : inputsym,
        "arg_operator_id" : self.opid,
        "arg_pf" : pf
      }

class MyriaShuffleConsumer(algebra.UnaryOperator, MyriaOperator):
  """A Myria ShuffleConsumer"""
  def __init__(self, input, opid):
    algebra.UnaryOperator.__init__(self, input)
    self.opid = opid

  def compileme(self, resultsym, inputsym):
    return {
        'op_name' : resultsym,
        'op_type' : 'ShuffleConsumer',
        'arg_operator_id' : self.opid,
      }

class BreakShuffle(rules.Rule):
  def fire(self, expr):
    if not isinstance(expr, MyriaShuffle):
      return expr

    opid = gen_op_id()
    producer = MyriaShuffleProducer(expr.input, opid, expr.columnlist)
    consumer = MyriaShuffleConsumer(producer, opid)
    return consumer

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

class ShuffleBeforeJoin(rules.Rule):
  def fire(self, expr):
    # If not a join, who cares?
    if not (isinstance(expr, algebra.Join) or \
            isinstance(expr, algebra.ProjectingJoin)):
      return expr

    # If both have shuffles already, who cares?
    if isinstance(expr.left, algebra.Shuffle) and isinstance(expr.right, algebra.Shuffle):
      return expr

    # Figure out which columns go in the shuffle
    left_cols, right_cols = MyriaLocalJoin.convertcondition(expr.condition)

    # Left shuffle
    if isinstance(expr.left, algebra.Shuffle):
      left_shuffle = expr.left
    else:
      left_shuffle = algebra.Shuffle(expr.left, left_cols)
    # Right shuffle
    if isinstance(expr.right, algebra.Shuffle):
      right_shuffle = expr.right
    else:
      right_shuffle = algebra.Shuffle(expr.right, right_cols)

    # Construct the object!
    if isinstance(expr, algebra.ProjectingJoin):
      return algebra.ProjectingJoin(expr.condition, left_shuffle, right_shuffle, expr.columnlist)
    elif isinstance(expr, algebra.Join):
      return algebra.Join(expr.condition, left_shuffle, right_shuffle)
    raise NotImplementedError("How the heck did you get here?")

class MyriaAlgebra:
  language = MyriaLanguage

  operators = [
      MyriaLocalJoin
      , MyriaSelect
      , MyriaProject
      , MyriaSQLiteScan
  ]

  rules = [
      rules.ProjectingJoin()
      , ShuffleBeforeJoin()
      , rules.OneToOne(algebra.Store,MyriaSQLiteInsert)
      , rules.OneToOne(algebra.Join,MyriaLocalJoin)
      , rules.OneToOne(algebra.Select,MyriaSelect)
      , rules.OneToOne(algebra.Shuffle,MyriaShuffle)
      , rules.OneToOne(algebra.Project,MyriaProject)
      , rules.OneToOne(algebra.ProjectingJoin,MyriaLocalJoin)
      , rules.OneToOne(algebra.Scan,MyriaSQLiteScan)
      , BreakShuffle()
      #, Parallel(2)
  ]
