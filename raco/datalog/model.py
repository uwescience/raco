'''
Classes for representing and manipulating Datalog programs.

In particular, they can be compiled to (iterative) relational algebra expressions.
'''
import networkx as nx
import raco.boolean
import raco.algebra
import raco.catalog

class Program:
  def __init__(self, rules):
    self.rules = rules

  def IDB(self,term):
    """Return a list of rules that define an IDB corresponding to the given term: relation names are the same, and the number of columns are the same."""
    matches = []
    for r in self.rules:
      if r.IDBof(term):
        matches.append(r)

    return matches

  def __repr__(self):
    return "\n".join([str(r) for r in self.rules])

class JoinSequence:
  """Convenience class for operating on a sequence of joins.  
A planner takes a joingraph and emits a join sequence.  
It's not yet a relational plan; we still need to do some bookkeeping 
with respect to datalog terms"""
 
  def __init__(self):
    self.terms = []
    self.conditions = []

  def offset(self, term):
    """Find the position offset for columns in term.  
For example, offset([A(X,Y), B(Y,Z)], B) == 2 
and offset([A(X,Y), B(Y,Z)], A) == 0"""
    pos = 0
    for t in self.terms:
      if t == term:
        return pos
      pos = pos + len(t.valuerefs)

    raise KeyError("%s not found in join sequence %s" % (term, self.terms))
      
  def makePlan(self, selection_conditions, program):
    """Make a relational plan. The selection_conditions are explicit conditions from the Datalog rule, like X=3"""
    leaves = [t.makeLeaf(selection_conditions, program) for t in self.terms]

    if not leaves: return raco.algebra.EmptyRelation()

    leftmost = leaves[0]
    pairs = zip(self.conditions, leaves[1:])

    def makejoin(leftplan, (condition, right)): 
      return raco.algebra.Join(condition, leftplan, right)

    return reduce(makejoin, pairs, leftmost)

  def addjoin(self, left, right, condition):
    """Append a join to the sequence.  The left term should already appear somewhere in the sequence."""
    if not self.terms:
      self.terms.append(left)

    leftoffset = self.offset(left)
    condition.leftoffset(leftoffset)
    self.terms.append(right)
    self.conditions.append(condition)

  def __repr__(self):
    if not self.terms:
      return "EmptyJoinSequence"
    else:
      return " ".join(["%s *%s " % (t,c) for t,c in zip(self.terms, self.conditions)]) + "%s" % self.terms[-1]
 
  def __iter__(self): 
    return self.terms.__iter__()

class Planner:
  """Given a join graph, produce a join sequence by choosing a join order with respect to 
some cost function.  Subclasses can implement this however they want."""

  def __init__(self, joingraph):
    self.joingraph = joingraph

  def joininfo(self, joinedge):
    """Given a join edge, return a pair of terms and the metadata for the join"""
    # TODO: Does this belong here, or in JoinSequence?  Especially the flip logic.
    leftterm, rightterm = joinedge
    data = self.joingraph.get_edge_data(*joinedge)
    condition = data["condition"]
    #print joinedge, condition
    left, right = data["order"]
    # We may have traversed the graph in the opposite direction
    # if so, flip the condition 
    if leftterm != left:
      condition = condition.flip()
    return leftterm, rightterm, condition

  def toJoinSequence(self, edgesequence, joinsequence=None):
    """Convert a sequence of join edges to a JoinSequence object.
A joinsequence has the correct offsets and orientation for positional references.
An edgesequence does not."""

    if joinsequence is None: joinsequence = JoinSequence()

    # TODO: Move this to JoinSequence?
    if not edgesequence:
      return joinsequence

    left, right, condition = self.joininfo(edgesequence[0])
      
    joinsequence.addjoin(left, right, condition)

    return self.toJoinSequence(edgesequence[1:], joinsequence)


def normalize(x,y):
  if y < x:
    edge = (y,x)
  else:  
    edge = (x,y)
  return edge

class BFSLeftDeepPlanner(Planner):

  def chooseplan(self, costfunc=None):
    """Return a join sequence object based on the join graph.
This one is simple -- it just adds the joins according to a breadth first search"""
    # choose first node
    firstnode = self.joingraph.nodes()[0]

    # get a BFS ordering of the edges.  Ignores costs.
    edgesequence = [x for x in nx.bfs_edges(self.joingraph, firstnode)]

    # Generate a concrete sequence of terms with conditions properly adjusted
    joinsequence = self.toJoinSequence(edgesequence)
    return joinsequence

class Rule:
  def __init__(self, headbody):
    self.head = headbody[0]
    self.body = headbody[1]
    # flag used to etect recursion
    self.compiling = False

  def IDBof(self, term):
    """Return true if this rule defines an IDB corresponding to the gien term"""
    return term.name == self.head.name and len(term.valuerefs) == len(self.head.valuerefs)
      
  def toRA(self, program):
    """Emit a relational plan for this rule"""
    if self.compiling:
      raise ValueError("Recursion not implemented")
    else:
      self.compiling = True

    newterms = []

    # get the terms, like A(X,Y,"foo")
    terms = [c for c in self.body if isinstance(c, Term)]

    # get the conditions, like Z=3
    conditions = [c for c in self.body if isinstance(c, raco.boolean.BinaryBooleanOperator)]

    joingraph = nx.Graph()
    N = len(terms)
    for i, term1 in enumerate(terms):
      joingraph.add_node(term1, term=term1) 
      for j in range(i+1, N):
        term2 = terms[j]
        joins = term1.joins(term2, conditions)
        if joins:
          conjunction = reduce(raco.boolean.AND, joins) 
          joingraph.add_edge(term1, term2, condition=conjunction, order=(term1, term2)) 

    comps = nx.connected_component_subgraphs(joingraph)

    component_plans = []

    for component in comps:
      # TODO: Only handling one component right now
      cycleconditions = []
      for cycle in nx.cycle_basis(component):
        # choose an edge to break the cycle
        # that one will be a selection condition on the final join
        oneedge = cycle[-2:]
        data = component.get_edge_data(*oneedge)   
        cycleconditions.append(data)
        component.remove_edge(*oneedge)

      if len(component) == 1:
        # no joins to plan
        onlyterm = component.nodes()[0]
        plan = onlyterm.makeLeaf(conditions, program)
      else:
        # TODO: clean this up. joingraph -> joinsequence -> relational plan
        planner = BFSLeftDeepPlanner(component)

        joinsequence = planner.chooseplan() 
      
        # create a relational plan, finally
        # pass in the conditions to make the leaves of the plan
        plan = joinsequence.makePlan(conditions, program)

        
      for condition_info in cycleconditions:
        predicate = condition_info["condition"]
        order = condition_info["order"]
 
        leftoffset = joinsequence.offset(order[0])
        rightoffset = joinsequence.offset(order[1])

        predicate.leftoffset(leftoffset)
        predicate.rightoffset(rightoffset)

        plan = raco.algebra.Select(predicate, plan)

      component_plans.append(plan)
    
    # link the components with a cross product    
    plan = component_plans[0]
    for newplan in component_plans[1:]:
      plan = raco.algebra.CrossProduct(plan, newplan)
 
    # Put a project on the end

    vars = [(i,nm) for i, (nm, typ) in enumerate(plan.scheme())]
    Pos = raco.boolean.PositionReference
    def findvar(var):
      occurrences = [Pos(i) for (i, nm) in vars if nm == var.var]
      if not occurrences:
        msg = "Head variable %s does not appear in rule body: %s" % (var, self)
        raise SyntaxError(msg)
      return occurrences[0]
      
    columnlist = [findvar(var) for var in self.head.valuerefs if isinstance(var, Var)]
        
    plan = raco.algebra.Project(columnlist, plan)

    self.compiling = False
    return plan      

  def __repr__(self):
    return "%s :- %s" % (self.head, ", ".join([str(t) for t in self.body]))

class Var:
  def __init__(self, var):
    self.var = var

  def __repr__(self):
    return str(self.var)

class Term:
  def __init__(self, parsedterm): 
    self.name = parsedterm[0]
    self.valuerefs = [vr for vr in parsedterm[1]]

  def __repr__(self):
    return "%s_%s(%s)" % (self.__class__.__name__,self.name, ",".join([str(e) for e in self.valuerefs]))

  def vars(self):
    """Return a dictionary mapping variable names to RA attribute references. For example, A(X,Y) returns {"X":PositionReference(0), "Y":PositionReference(1)}.  Only the first reference to this variable is returned."""
  
    Pos = raco.boolean.PositionReference
    return {vr.var:Pos(i) for i,vr in enumerate(self.valuerefs) if isinstance(vr, Var)}

  def joins(self, other, conditions):
    """Return the join conditions between this term and the argument term.  The second argument is a list of explicit conditions, like X=3 or Y=Z"""
    # get the implicit join conditions
    yourvars = other.vars()
    myvars = self.vars()
    joins = []
    for var,attr in myvars.items():
      if yourvars.has_key(var):
        joins.append(raco.boolean.EQ(attr, yourvars[var]))

    for c in conditions:
      if isinstance(c.left, Var) and isinstance(c.right, Var):
        # then we have a potential join condition
        if self.match(c.left) and other.match(c.right):
          joins.append(c.__class__(self.convertvalref(c.left), other.convertvalref(c.right)))
        if other.match(c.left) and self.match(c.right):
          joins.append(c.__class__(self.convertvalref(c.left), other.convertvalref(c.right)))

    return joins
        
  def match(self, valref):
    """Return true if valref is a variable and is used in the list"""
    return isinstance(valref, Var) and valref.var in self.vars()

  def position(self, valueref):
    """ Returns the position of the variable in the term.  Throws an error if it does not appear."""
    return [v for v in self.vars()].index(valueref.var)

  def convertvalref(self, valueref):
    """Convert a Datalog value reference (a literal or a variable) to RA.  Literals are passed through unchanged.  Variables are converted"""
    if self.match(valueref):
      pos = self.position(valueref)
      return raco.boolean.PositionReference(pos)
    else:
      # must be a join condition, and the other variable matches
      return valueref

  def explicitconditions(self, conditions):
    """Given a list of parsed Datalog boolean conditions, return an iterator over RA selection conditions that apply to this term.  Ignore join conditions."""
    # This method is written rather verbosely for clarity

    Literal = raco.boolean.Literal

    for condition in conditions:
    
      if isinstance(condition.left,Literal) and isinstance(condition.right,Literal):
        # No variables
        yield condition.__class__(condition.left, condition.right)

      elif isinstance(condition.left,Literal) and isinstance(condition.right, Var):
        # left literal, right variable, like 3 = X
        if self.match(condition.right):
          yield condition.__class__(condition.left, self.convertvalref(condition.right))
        else:
          # variable is not used in this term, so do nothing
          pass

      elif isinstance(condition.left, Var) and isinstance(condition.right, Literal):
        # left variable, right literal, like X = 3
        if self.match(condition.left):
          yield condition.__class__(self.convertvalref(condition.left), condition.right)
        else:
          # variable is not used in this term, so do nothing
          pass

      elif isinstance(condition.left, Var) and isinstance(condition.right, Var):
        # two variables. Both match, neither match, or one matches. 
        leftmatch = self.match(condition.left)
        rightmatch = self.match(condition.right)
        if leftmatch and rightmatch:
          # a selection condition like A(X,Y), X=Y
          yield condition.__class__(self.convertvalref(condition.left), self.convertvalref(condition.right))
        elif not leftmatch and not rightmatch:
          # nothing to do with this term
          pass
        elif not leftmatch and rightmatch:
          # join condition, leave it alone 
          pass
        elif leftmatch and not rightmatch:
          # join condition, leave it alone 
          pass
        
  def implicitconditions(self):
    """An iterator over implicit selection conditions derived from the datalog syntax.
For example, A(X,X) implies position0 == position1, and A(X,4) implies position1 == 4"""
    # Check for implicit literal equality conditions, like A(X,"foo")
    for i,b in enumerate(self.valuerefs):
      if isinstance(b,raco.boolean.Literal):
        posref = raco.boolean.PositionReference(i)
        yield raco.boolean.EQ(posref, b)

    # Check for repeated variable conditions, like A(X,X)
    N = len(self.valuerefs)
    for i,x in enumerate(self.valuerefs):
      if isinstance(x, Var):
        for j in range(i + 1, N):
          y = self.valuerefs[j]
          if isinstance(y, Var):
            # TODO: probably want to implement __eq__, but it makes Var objects unhashable
            if x.var == y.var:
              leftpos = raco.boolean.PositionReference(i)
              rightpos = raco.boolean.PositionReference(j)
              yield raco.boolean.EQ(leftpos, rightpos)
 
  def makeLeaf(term, conditions, program):
    """Return an RA plan that Scans the appropriate relation and applies all selection conditions
  Two sources of conditions: the term itself, like A(X,"foo") -> Select(pos1="foo", Scan(A))
  and separate condition terms, like A(X,Y), X=3 -> Select(pos0=3, Scan(A))
  separate condition terms are passed in as an argument.
  """
    def attr(i,r): 
      if isinstance(r,Var):
        return (r.var, None)
      elif isinstance(r,raco.boolean.Literal):
        return ("pos%s" % i, type(r.value))

    idbs = program.IDB(term)
    if idbs:
      scan = reduce(raco.algebra.Union,[idb.toRA(program) for idb in idbs])
    else:
      scheme = [attr(i,r) for i,r in enumerate(term.valuerefs)]
      scan = raco.algebra.Scan(raco.catalog.Relation(term.name, scheme))

    # collect conditions within the term itself, like A(X,3) or A(Y,Y)
    implconds = list(term.implicitconditions())

    # collect explicit conditions, like A(X,Y), Y=2
    explconds = list(term.explicitconditions(conditions))
   
    allconditions = implconds + explconds

    if allconditions:
      conjunction = reduce(raco.boolean.AND, allconditions)
      plan = raco.algebra.Select(conjunction, scan)
    else:
      plan = scan  # TODO: This is only correct for EDBs

    plan.set_alias(term)
    return plan

class IDB(Term):
  def __init__(self, termobj):
    self.name = termobj.name
    self.valuerefs = termobj.valuerefs

class EDB(Term):
  pass

