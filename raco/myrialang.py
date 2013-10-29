# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
#
# The above modeline for Vim automatically sets it to
# .. Bill's Python style. If it doesn't work, check
#     :set modeline?        -> should be true
#     :set modelines?       -> should be > 0

import algebra
import boolean
import json
from operator import and_
import rules
import scheme
import sys
import expression
from language import Language
from utility import emit

def scheme_to_schema(s):
  if s:
    names, descrs = zip(*s.asdict.items())
    names = ["%s" % n for n in names]
    types = [r[1] for r in descrs]
  else:
    names = []
    types = []
  return {"column_types" : types, "column_names" : names}

def resolve_relation_key(key):
  """Extract user, program, relation strings from a colon-delimited string.

  User and program can be omitted, in which case the system chooses default
  values."""

  toks = key.split(':')

  user = 'public'
  program = 'adhoc'
  relation = toks[-1]

  try:
    program = toks[-2]
    user = toks[-3]
  except IndexError:
    pass

  return user, program, relation

def compile_expr(op, child_scheme):
  ####
  # Put special handling at the top!
  ####
  if isinstance(op, expression.NumericLiteral) or isinstance(op, boolean.NumericLiteral):
    if type(op.value) == int:
      if op.value <= 2**31-1 and op.value >= -2**31:
        myria_type = 'INT_TYPE'
      else:
        myria_type = 'LONG_TYPE'
    elif type(op.value) == float:
      myria_type = 'DOUBLE_TYPE'
    else:
      raise NotImplementedError("Compiling NumericLiteral %s of type %s" % (op, type(op.value)))

    return {
        'type' : 'CONSTANT',
        'value' : str(op.value),
        'value_type' : myria_type
    }
  elif isinstance(op, expression.AttributeRef):
    return {
        'type' : 'VARIABLE',
        'column_idx' : op.get_position(child_scheme)
    }
  ####
  # Everything below here is compiled automatically
  ####
  elif isinstance(op, expression.UnaryOperator):
    return {
        'type' : op.opname(),
        'operand' : compile_expr(op.input, child_scheme)
    }
  elif isinstance(op, expression.BinaryOperator) or isinstance(op, boolean.BinaryBooleanOperator):
    return {
        'type' : op.opname(),
        'left' : compile_expr(op.left, child_scheme),
        'right' : compile_expr(op.right, child_scheme)
    }
  raise NotImplementedError("Compiling expr of class %s" % op.__class__)

def compile_mapping(expr, child_scheme):
  output_name, root_op = expr
  return {
      'output_name' : output_name,
      'root_expression_operator' : compile_expr(root_op, child_scheme)
  }

class MyriaLanguage(Language):
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
  language = MyriaLanguage

class MyriaScan(algebra.Scan, MyriaOperator):
  def compileme(self, resultsym):
    user, program, relation = resolve_relation_key(self.relation_key)

    return {
        "op_name" : resultsym,
        "op_type" : "TableScan",
        "relation_key" : {
          "user_name" : user,
          "program_name" : program,
          "relation_name" : relation
        }
      }

class MyriaScanTemp(algebra.ScanTemp, MyriaOperator):
  def compileme(self, resultsym):
    user, program, relation = resolve_relation_key(self.name)

    return {
        "op_name" : resultsym,
        "op_type" : "TableScan",
        "relation_key" : {
          "user_name" : user,
          "program_name" : program,
          "relation_name" : relation
        }
      }

class MyriaUnionAll(algebra.UnionAll, MyriaOperator):
  def compileme(self, resultsym, leftsym, rightsym):
    return {
        "op_name" : resultsym,
        "op_type" : "UnionAll",
        "arg_children" : [leftsym, rightsym]
      }

class MyriaSingleton(algebra.SingletonRelation, MyriaOperator):
  def compileme(self, resultsym):
    return {
        "op_name" : resultsym,
        "op_type" : "Singleton",
      }

class MyriaEmptyRelation(algebra.EmptyRelation, MyriaOperator):
  def compileme(self, resultsym):
    return {
        "op_name" : resultsym,
        "op_type" : "Empty",
        'arg_schema' : scheme_to_schema(self.scheme())
        }

class MyriaSelect(algebra.Select, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    pred = compile_expr(self.condition, self.scheme())
    return {
      "op_name" : resultsym,
      "op_type" : "Filter",
      "arg_child" : inputsym,
      "arg_predicate" : pred
    }

class MyriaCrossProduct(algebra.CrossProduct, MyriaOperator):
  def compileme(self, resultsym, leftsym, rightsym):
    column_names = [name for (name,type) in self.scheme()]
    allleft = [i.position for i in self.left.scheme().ascolumnlist()]
    allright = [i.position for i in self.right.scheme().ascolumnlist()]
    return {
        "op_name" : resultsym,
        "op_type" : "SymmetricHashJoin",
        "arg_column_names" : column_names,
        "arg_child1" : leftsym,
        "arg_child2" : rightsym,
        "arg_columns1" : [],
        "arg_columns2" : [],
        "arg_select1" : allleft,
        "arg_select2" : allright
      }

class MyriaStore(algebra.Store, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    user, program, relation = resolve_relation_key(self.relation_key)

    return {
        "op_name" : resultsym,
        "op_type" : "DbInsert",
        "relation_key" : {
          "user_name" : user,
          "program_name" : program,
          "relation_name" : relation
        },
        "arg_overwrite_table" : True,
        "arg_child" : inputsym,
      }

class MyriaStoreTemp(algebra.StoreTemp, MyriaOperator):
  def compileme(self, resultsym, inputsym):
    user, program, relation = resolve_relation_key(self.name)

    return {
        "op_name" : resultsym,
        "op_type" : "DbInsert",
        "relation_key" : {
          "user_name" : user,
          "program_name" : program,
          "relation_name" : relation
        },
        "arg_overwrite_table" : True,
        "arg_child" : inputsym,
      }

class MyriaSymmetricHashJoin(algebra.ProjectingJoin, MyriaOperator):
  @classmethod
  def convertcondition(self, condition):
    """Convert the joincondition to a list of left columns and a list of right columns representing a conjunction"""


    if isinstance(condition, boolean.AND) or isinstance(condition, expression.AND):
      leftcols1, rightcols1 = self.convertcondition(condition.left)
      leftcols2, rightcols2 = self.convertcondition(condition.right)
      return leftcols1 + leftcols2, rightcols1 + rightcols2

    if isinstance(condition, boolean.EQ) or isinstance(condition, expression.EQ):
      return [condition.left.position], [condition.right.position]

    raise NotImplementedError("Myria only supports EquiJoins, not %s" % condition)
  
  def compileme(self, resultsym, leftsym, rightsym):
    """Compile the operator to a sequence of json operators"""
  
    leftcols, rightcols = self.convertcondition(self.condition)

    if self.columnlist is None:
      self.columnlist = self.scheme().ascolumnlist()
    column_names = [name for (name,type) in self.scheme()]

    allleft = [i.position for i in self.columnlist if i.position < len(self.left.scheme())]
    allright = [i.position-len(self.left.scheme()) for i in self.columnlist if i.position >= len(self.left.scheme())]

    join = {
        "op_name" : resultsym,
        "op_type" : "SymmetricHashJoin",
        "arg_column_names" : column_names,
        "arg_child1" : "%s" % leftsym,
        "arg_columns1" : leftcols,
        "arg_child2": "%s" % rightsym,
        "arg_columns2" : rightcols,
        "arg_select1" : allleft,
        "arg_select2" : allright
      }

    return join

class MyriaGroupBy(algebra.GroupBy, MyriaOperator):
  @staticmethod
  def agg_mapping(agg_expr):
    """Maps an AggregateExpression to a Myria string constant representing the
    corresponding aggregate operation."""
    if isinstance(agg_expr, expression.MAX):
      return "AGG_OP_MAX"
    elif isinstance(agg_expr, expression.MIN):
      return "AGG_OP_MIN"
    elif isinstance(agg_expr, expression.COUNT):
      return "AGG_OP_COUNT"
    elif isinstance(agg_expr, expression.SUM):
      return "AGG_OP_SUM"

  def compileme(self, resultsym, inputsym):
    child_scheme = self.input.scheme()
    group_fields = [expression.toUnnamed(ref, child_scheme) \
                    for ref in self.groupinglist]
    agg_fields = [expression.toUnnamed(expr.input, child_scheme) \
                  for expr in self.aggregatelist]
    agg_types = [[MyriaGroupBy.agg_mapping(agg_expr)] \
                 for agg_expr in self.aggregatelist]
    ret = {
        "op_name" : resultsym,
        "arg_child" : inputsym,
        "arg_agg_fields" : [agg_field.position for agg_field in agg_fields],
        "arg_agg_operators" : agg_types,
        }

    num_fields = len(self.groupinglist)
    if num_fields == 0:
      ret["op_type"] = "Aggregate"
    elif num_fields == 1:
      ret["op_type"] = "SingleGroupByAggregate"
      ret["arg_group_field"] = group_fields[0].position
    else:
      ret["op_type"] = "MultiGroupByAggregate"
      ret["arg_group_fields"] = [field.position for field in group_fields]
    return ret

class MyriaShuffle(algebra.Shuffle, MyriaOperator):
  """Represents a simple shuffle operator"""
  def compileme(self, resultsym, inputsym):
    raise NotImplementedError('shouldn''t ever get here, should be turned into SP-SC pair')

class MyriaCollect(algebra.Collect, MyriaOperator):
  """Represents a simple collect operator"""
  def compileme(self, resultsym, inputsym):
    raise NotImplementedError('shouldn''t ever get here, should be turned into CP-CC pair')

class MyriaDupElim(algebra.Distinct, MyriaOperator):
  """Represents duplicate elimination"""
  def compileme(self, resultsym, inputsym):
    return {
        "op_name" : resultsym,
        "op_type" : "DupElim",
        "arg_child" : inputsym,
    }


class MyriaApply(algebra.Apply, MyriaOperator):
  """Represents a simple apply operator"""
  def compileme(self, resultsym, inputsym):
    child_scheme = self.input.scheme()
    exprs = [compile_mapping(x, child_scheme) for x in self.mappings]
    return {
        'type' : 'Apply',
        'name' : resultsym,
        'arg_child' : inputsym,
        'expressions' : exprs
    }

class MyriaBroadcastProducer(algebra.UnaryOperator, MyriaOperator):
  """A Myria BroadcastProducer"""
  def __init__(self, input):
    algebra.UnaryOperator.__init__(self, input)

  def shortStr(self):
    return "%s" % self.opname()

  def compileme(self, resultsym, inputsym):
    return {
        "op_name" : resultsym,
        "op_type" : "BroadcastProducer",
        "arg_child" : inputsym,
      }

class MyriaBroadcastConsumer(algebra.UnaryOperator, MyriaOperator):
  """A Myria BroadcastConsumer"""
  def __init__(self, input):
    algebra.UnaryOperator.__init__(self, input)

  def shortStr(self):
    return "%s" % self.opname()

  def compileme(self, resultsym, inputsym):
    return {
        'op_name' : resultsym,
        'op_type' : 'BroadcastConsumer',
        'arg_child' : inputsym,
        'arg_schema' : scheme_to_schema(self.scheme())
      }

class MyriaShuffleProducer(algebra.UnaryOperator, MyriaOperator):
  """A Myria ShuffleProducer"""
  def __init__(self, input, hash_columns):
    algebra.UnaryOperator.__init__(self, input)
    self.hash_columns = hash_columns

  def shortStr(self):
    hash_string = ','.join([str(x) for x in self.hash_columns])
    return "%s(h(%s))" % (self.opname(), hash_string)

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
        "arg_pf" : pf
      }

class MyriaShuffleConsumer(algebra.UnaryOperator, MyriaOperator):
  """A Myria ShuffleConsumer"""
  def __init__(self, input):
    algebra.UnaryOperator.__init__(self, input)

  def shortStr(self):
    return "%s" % self.opname()

  def compileme(self, resultsym, inputsym):
    return {
        'op_name' : resultsym,
        'op_type' : 'ShuffleConsumer',
        'arg_child' : inputsym,
        'arg_schema' : scheme_to_schema(self.scheme())
      }

class BreakShuffle(rules.Rule):
  def fire(self, expr):
    if not isinstance(expr, MyriaShuffle):
      return expr

    producer = MyriaShuffleProducer(expr.input, expr.columnlist)
    consumer = MyriaShuffleConsumer(producer)
    return consumer


class MyriaCollectProducer(algebra.UnaryOperator, MyriaOperator):
  """A Myria CollectProducer"""
  def __init__(self, input, server):
    algebra.UnaryOperator.__init__(self, input)
    self.server = server

  def shortStr(self):
    return "%s(@%s)" % (self.opname(), self.server)

  def compileme(self, resultsym, inputsym):
    return {
        "op_name" : resultsym,
        "op_type" : "CollectProducer",
        "arg_child" : inputsym,
      }

class MyriaCollectConsumer(algebra.UnaryOperator, MyriaOperator):
  """A Myria CollectConsumer"""
  def __init__(self, input):
    algebra.UnaryOperator.__init__(self, input)

  def shortStr(self):
    return "%s" % self.opname()

  def compileme(self, resultsym, inputsym):
    return {
        'op_name' : resultsym,
        'op_type' : 'CollectConsumer',
        'arg_child' : self.inputsym,
        'arg_schema' : scheme_to_schema(self.scheme())
      }

class BreakCollect(rules.Rule):
  def fire(self, expr):
    if not isinstance(expr, MyriaCollect):
      return expr

    producer = MyriaCollectProducer(expr.input, None)
    consumer = MyriaCollectConsumer(producer)
    return consumer

class BreakBroadcast(rules.Rule):
  def fire(self, expr):
    if not isinstance(expr, algebra.Broadcast):
      return expr

    producer = MyriaBroadcastProducer(expr.input)
    consumer = MyriaBroadcastConsumer(producer)
    return consumer

class ShuffleBeforeJoin(rules.Rule):
  def fire(self, expr):
    # If not a join, who cares?
    if not isinstance(expr, algebra.Join):
      return expr

    # If both have shuffles already, who cares?
    if isinstance(expr.left, algebra.Shuffle) and isinstance(expr.right, algebra.Shuffle):
      return expr

    # Convert to unnamed perspective
    condition = MyriaLanguage.unnamed(expr.condition, expr.scheme())

    # Figure out which columns go in the shuffle
    left_cols, right_cols = MyriaSymmetricHashJoin.convertcondition(expr.condition)

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

class BroadcastBeforeCross(rules.Rule):
  def fire(self, expr):
    # If not a CrossProduct, who cares?
    if not isinstance(expr, algebra.CrossProduct):
      return expr

    if isinstance(expr.left, algebra.Broadcast) or \
        isinstance(expr.right, algebra.Broadcast):
      return expr

    # By default, broadcast the right child
    expr.right = algebra.Broadcast(expr.right)

    return expr

class RemoveInnerStores(rules.Rule):
  is_root = True
  def fire(self, expr):
    # This rule only works because, currently, the compiler adds a MyriaStore
    # during compilation (and after this rule is fired).

    if not self.is_root and isinstance(expr, algebra.Store):
      return expr.input

    if self.is_root:
      self.is_root = False

    return expr

class TransferBeforeGroupBy(rules.Rule):
  def fire(self, expr):
    # If not a GroupBy, who cares?
    if not isinstance(expr, algebra.GroupBy):
      return expr

    # Get an array of position references to columns in the child scheme
    child_scheme = expr.input.scheme()
    group_fields = [expression.toUnnamed(ref, child_scheme).position \
                    for ref in expr.groupinglist]
    if len(group_fields) == 0:
      # Need to Collect all tuples at once place
      expr.input = algebra.Collect(expr.input)
    else:
      # Need to Shuffle
      expr.input = algebra.Shuffle(expr.input, group_fields)

    return expr

class SplitSelects(rules.Rule):
  """If a select has an AND, replace it with two consecutive selects."""
  def fire(self, expr):
    if isinstance(expr, algebra.Select):
      if isinstance(expr.condition, boolean.AND) or \
          isinstance(expr.condition, expression.AND):
        first_filter = algebra.Select(expr.condition.left, expr.input)
        return algebra.Select(expr.condition.right, first_filter)
    return expr

class ProjectToDistinctColumnSelect(rules.Rule):
  def fire(self, expr):
    # If not a Project, who cares?
    if not isinstance(expr, algebra.Project):
      return expr

    mappings = [(None, x) for x in expr.columnlist]
    colSelect = algebra.Apply(mappings, expr.input)
    # TODO(dhalperi) the distinct logic is broken because we don't have a
    # locality-aware optimizer. For now, don't insert Distinct for a logical
    # project. This is BROKEN.
    # distinct = algebra.Distinct(colSelect)
    # return distinct
    return colSelect

class SimpleGroupBy(rules.Rule):
  def fire(self, expr):
    if not isinstance(expr, algebra.GroupBy):
      return expr

    child_scheme = expr.input.scheme()
    def is_child_ref(agg):
      return isinstance(agg, expression.COUNTALL) or \
          isinstance(agg, expression.UnaryOperator) and isinstance(agg.input, expression.AttributeRef)

   # Complicated expressions are those that aggregate over something that is not a child_ref
    agg_expr_refs = [agg for agg in expr.aggregatelist if not is_child_ref(agg)]

    if len(agg_expr_refs) == 0:
      # There are no complicated expressions, we're okay.
      return expr

    agg_child_refs = [agg for agg in expr.aggregatelist if is_child_ref(agg)]

    # Let's construct the Apply operator instead, and update the agg list
    mappings = [(None, expression.UnnamedAttributeRef(i))
                for i in range(len(child_scheme))]
    for agg_expr in agg_expr_refs:
      mappings.append((None, agg_expr.input))
      agg_expr.input = expression.UnnamedAttributeRef(len(mappings)-1)
      agg_child_refs.append(agg_expr)

    new_apply = algebra.Apply(mappings, expr.input)
    expr.input = new_apply
    # Don't overwrite expr.aggregatelist, instead we are mutating the objects
    # it contains when we modify agg_expr in the above for loop.
    return expr

class DropTemps(rules.Rule):
  def fire(self, expr):
    if isinstance(expr, algebra.ScanTemp):
      return algebra.Scan(expr.name, expr._scheme)
    if isinstance(expr, algebra.StoreTemp):
      return algebra.Store(expr.name, expr.input)
    return expr

class MyriaAlgebra:
  language = MyriaLanguage

  operators = [
      MyriaSymmetricHashJoin
      , MyriaSelect
      , MyriaScan
      , MyriaStore
  ]

  fragment_leaves = (
      MyriaShuffleConsumer
      , MyriaCollectConsumer
      , MyriaBroadcastConsumer
      , MyriaScan
  )

  rules = [
#      DropTemps()
      rules.ProjectingJoin()
      , rules.JoinToProjectingJoin()
      , ShuffleBeforeJoin()
      , BroadcastBeforeCross()
      , TransferBeforeGroupBy()
#      , SplitSelects()
      , SimpleGroupBy()
      , ProjectToDistinctColumnSelect()
      , rules.OneToOne(algebra.CrossProduct,MyriaCrossProduct)
      , rules.OneToOne(algebra.Store,MyriaStore)
      , rules.OneToOne(algebra.StoreTemp,MyriaStoreTemp)
      , rules.OneToOne(algebra.Apply,MyriaApply)
      , rules.OneToOne(algebra.Select,MyriaSelect)
      , rules.OneToOne(algebra.GroupBy,MyriaGroupBy)
      , rules.OneToOne(algebra.Distinct,MyriaDupElim)
      , rules.OneToOne(algebra.Shuffle,MyriaShuffle)
      , rules.OneToOne(algebra.Collect,MyriaCollect)
      , rules.OneToOne(algebra.ProjectingJoin,MyriaSymmetricHashJoin)
      , rules.OneToOne(algebra.Scan,MyriaScan)
      , rules.OneToOne(algebra.ScanTemp,MyriaScanTemp)
      , rules.OneToOne(algebra.SingletonRelation,MyriaSingleton)
      , rules.OneToOne(algebra.EmptyRelation,MyriaEmptyRelation)
      , rules.OneToOne(algebra.UnionAll,MyriaUnionAll)
#      , RemoveInnerStores()
      , BreakShuffle()
      , BreakCollect()
      , BreakBroadcast()
  ]

def apply_schema_recursive(operator, catalog):
  """Given a catalog, which has a function get_scheme(string) to map a relation
  name to its scheme, update the schema for all scan operations that scan
  relations in the map."""

  # We found a scan, let's fill in its scheme
  if isinstance(operator, MyriaScan):
    rel_key = operator.relation_key
    rel_scheme = catalog.get_scheme(rel_key)
    if rel_scheme:
      # The Catalog has an entry for this relation
      if len(operator.scheme()) != len(rel_scheme):
        s = "scheme for %s (%d cols) does not match the catalog (%d cols)" % (
          rel_key, len(operator._scheme), len(rel_scheme))
        raise ValueError(s)
      operator._scheme = rel_scheme
    else:
      # The specified relation is not in the Catalog; replace its scheme's
      # types with "unknown".
      old_sch = operator.scheme()
      new_sch = [(old_sch.getName(i), "unknown") for i in range(len(old_sch))]
      operator._scheme = scheme.Scheme(new_sch)

  # Recurse through all children
  for child in operator.children():
    apply_schema_recursive(child, catalog)

  # Done
  return

class EmptyCatalog:
  @staticmethod
  def get_scheme(relation_name):
    return None

def compile_to_json(raw_query, logical_plan, physical_plan, catalog=None):
  """This function compiles a logical RA plan to the JSON suitable for
  submission to the Myria REST API server."""

  # No catalog supplied; create the empty catalog
  if catalog is None:
    catalog = EmptyCatalog()

  for (label, root_op) in physical_plan:
    apply_schema_recursive(root_op, catalog)

  # A dictionary mapping each object to a unique, object-dependent symbol.
  # Since we want this to be truly unique for each object instance, even if two
  # objects are equal, we use id(obj) as the key.
  syms = {}

  def one_fragment(rootOp):
      """Given an operator that is the root of a query fragment/plan, extract
      the operators in the fragment. Assembles a list cur_frag of the operators
      in the current fragment, in preorder from the root.
      
      This operator also assembles a queue of the discovered roots of later
      fragments, e.g., when there is a ShuffleProducer below. The list of
      operators that should be treated as fragment leaves is given by
      MyriaAlgebra.fragment_leaves. """

      # The current fragment starts with the current root
      cur_frag = [rootOp]
      # If necessary, assign a symbol to the root operator
      if id(rootOp) not in syms:
          syms[id(rootOp)] = algebra.gensym()
      # Initially, there are no new roots discovered below leaves of this
      # fragment.
      queue = []
      if isinstance(rootOp, MyriaAlgebra.fragment_leaves):
          # The current root operator is a fragment leaf, such as a
          # ShuffleProducer. Append its children to the queue of new roots.
          for child in rootOp.children():
              queue.append(child)
      else:
          # Otherwise, the children belong in this fragment. Recursively go
          # discover their fragments, including the queue of roots below their
          # children.
          for child in rootOp.children():
              (child_frag, child_queue) = one_fragment(child)
              # Add their fragment onto this fragment
              cur_frag += child_frag
              # Add their roots-of-next-fragments into our queue
              queue += child_queue
      return (cur_frag, queue)

  def fragments(rootOp):
      """Given the root of a query plan, recursively determine all the fragments
      in it."""
      # The queue of fragment roots. Initially, just the root of this query
      queue = [rootOp]
      ret = []
      while len(queue) > 0:
          # Get the next fragment root
          rootOp = queue.pop(0)
          # .. recursively learn the entire fragment, and any newly discovered
          # roots.
          (op_frag, op_queue) = one_fragment(rootOp)
          # .. Myria JSON expects the fragment operators in reverse order,
          # i.e., root at the bottom.
          ret.append(reversed(op_frag))
          # .. and collect the newly discovered fragment roots.
          queue.extend(op_queue)
      return ret

  def call_compile_me(op):
      "A shortcut to call the operator's compile_me function."
      opsym = syms[id(op)]
      childsyms = [syms[id(child)] for child in op.children()]
      if isinstance(op, algebra.ZeroaryOperator):
          return op.compileme(opsym)
      if isinstance(op, algebra.UnaryOperator):
          return op.compileme(opsym, childsyms[0])
      if isinstance(op, algebra.BinaryOperator):
          return op.compileme(opsym, childsyms[0], childsyms[1])
      if isinstance(op, algebra.NaryOperator):
          return op.compileme(opsym, childsyms)
      raise NotImplementedError("unable to handle operator of type "+type(op))

  # The actual code. all_frags collects up the fragments.
  all_frags = []
  # For each IDB, generate a plan that assembles all its fragments and stores
  # them back to a relation named (label).
  for (label, rootOp) in physical_plan:
      if isinstance(rootOp, algebra.Store):
          # If there is already a store (including MyriaStore) at the top, do
          # nothing.
          frag_root = rootOp
      else:
          # Otherwise, add an insert at the top to store this relation to a
          # table named (label).
          frag_root = MyriaStore(plan=rootOp, relation_key=label)
      # Make sure the root is in the symbol dictionary, but rather than using a
      # generated symbol use the IDB label.
      syms[id(frag_root)] = label
      # Determine the fragments.
      frags = fragments(frag_root)
      # Build the fragments.
      all_frags.extend([{'operators': [call_compile_me(op) for op in frag]} for frag in frags])
      # Clear out the symbol dictionary for the next IDB.
      syms.clear()

  # Assemble all the fragments into a single JSON query plan
  query = {
          'fragments' : all_frags,
          'raw_datalog' : raw_query,
          'logical_ra' : str(logical_plan)
          }
  return query
