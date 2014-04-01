
# TODO: To be refactored into parallel shared memory lang,
# where you plugin in the parallel shared memory language specific codegen

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

template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grappa_templates")

def readtemplate(fname):
    return file(os.path.join(template_path, fname)).read()


base_template = readtemplate("base_query.template")

class GrappaStagedTupleRef(StagedTupleRef):
  def __afterDefinitionCode__(self):
     # Grappa requires structures to be block aligned if they will be
     # iterated over with localizing forall
     return "GRAPPA_BLOCK_ALIGNED"
    

class GrappaLanguage(Language):
    @classmethod
    def new_relation_assignment(cls, rvar, val):
        return """
    %s
    %s
    """ % (cls.relation_decl(rvar), cls.assignment(rvar, val))

    @classmethod
    def relation_decl(cls, rvar):
        return "GlobalAddress<Tuple> %s;" % rvar

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
    def log(txt):
        return  """LOG(INFO) << "%s";\n""" % txt
    
    @staticmethod
    def log_unquoted(code, level=0):
        if level == 0:
            log_str = "LOG(INFO)"
        else:
            log_str = "VLOG(%s)" % (level)

        return """%(log_str)s << %(code)s;\n""" % locals()

    @staticmethod
    def pipeline_wrap(ident, code, attrs):
        timer_metric = None
        if attrs['type'] == 'in_memory':
            timer_metric = "in_memory_runtime"
        elif attrs['type'] == 'scan':
            timer_metric = "scan_runtime"

        pipeline_template = """
        auto start_%(ident)s = walltime();
        %(code)s
        auto end_%(ident)s = walltime();
        auto runtime_%(ident)s = end_%(ident)s - start_%(ident)s;
        %(timer_metric)s += runtime_%(ident)s;
        VLOG(1) << "pipeline %(ident)s: " << runtime_%(ident)s << " s";
        """

        return pipeline_template % locals()


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
        decl = """int64_t %s;""" % (sid)
        init = """auto l_%(sid)s = string_index.string_lookup("%(s)s");
                   on_all_cores([=] { %(sid)s = l_%(sid)s; });""" % locals()
        return """(%s)""" % sid, [decl], [init]
        #raise ValueError("String Literals not supported in C language: %s" % s)

    @classmethod
    def negation(cls, input):
        innerexpr, decls, inits = input
        return "(!%s)" % (innerexpr,), decls, inits

    @classmethod
    def boolean_combine(cls, args, operator="&&"):
        opstr = " %s " % operator
        conjunc =        opstr.join(["(%s)" % arg for arg, _, _ in args])
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
            return '%s.get(%s)' % (symbol, position), [], []

class GrappaOperator (Pipelined):
    language = GrappaLanguage

from algebra import ZeroaryOperator
class MemoryScan(algebra.Scan, GrappaOperator):
  def produce(self, state):
    code = ""
    #generate the materialization from file into memory
    #TODO split the file scan apart from this in the physical plan

    # generate code for actual IO and materialization in memory
    fs = GrappaFileScan(self.relation_key, self._scheme)

    # Common subexpression elimination
    # don't scan the same file twice
    resultsym = state.lookupExpr(fs)
    LOG.debug("lookup %s(h=%s) => %s", fs, fs.__hash__(), resultsym)
    if not resultsym:
        #TODO for now this will break whatever relies on self.bound like reusescans
        #Scan is the only place where a relation is declared
        resultsym = gensym()

        fscode = fs.compileme(resultsym)
        state.saveExpr(fs, resultsym)

    # now generate the scan from memory
    inputsym = resultsym


# scan from index
#    memory_scan_template = """forall_localized( %(inputsym)s_index->vs, %(inputsym)s_index->nv, [](int64_t ai, Vertex& a) {
#      forall_here_async<&impl::local_gce>( 0, a.nadj, [=](int64_t start, int64_t iters) {
#      for (int64_t i=start; i<start+iters; i++) {
#        auto %(tuple_name)s = a.local_adj[i];
#          
#          %(inner_plan_compiled)s
#       } // end scan over %(inputsym)s (for)
#       }); // end scan over %(inputsym)s (forall_here_async)
#       }); // end scan over %(inputsym)s (forall_localized)
#       """

    memory_scan_template = """
forall( %(inputsym)s.data, %(inputsym)s.numtuples, [=](int64_t i, %(tuple_type)s& %(tuple_name)s) {
%(inner_plan_compiled)s
}); // end  scan over %(inputsym)s
"""

    rel_decl_template = """Relation<%(tuple_type)s> %(resultsym)s;"""

    stagedTuple = state.lookupTupleDef(inputsym)
    if not stagedTuple: # not subsumed by addDeclarations set, because StagedTupleRef.__init__ generates a new name
        stagedTuple = GrappaStagedTupleRef(inputsym, self.scheme())
        state.saveTupleDef(inputsym, stagedTuple)

        tuple_type_def = stagedTuple.generateDefinition()
        tuple_type = stagedTuple.getTupleTypename()

        # generate declaration of the in-memory relation
        rel_decl = rel_decl_template % locals()
        state.addDeclarations([tuple_type_def, rel_decl])

        # now that we have the type, format this in;
        state.addPipeline(fscode%{"result_type": tuple_type}, "scan")


    tuple_type = stagedTuple.getTupleTypename()
    tuple_name = stagedTuple.name

    inner_plan_compiled = self.parent.consume(stagedTuple, self, state)

    code += memory_scan_template % locals()
    state.addPipeline(code, "in_memory")
    
  def consume(self, t, src, state):
    assert False, "as a source, no need for consume"

  def __eq__(self, other):
      """
      For what we are using MemoryScan for, the only use
      of __eq__ is in hashtable lookups for CSE optimization.
      We omit self.schema because the relation_key determines
      the level of equality needed.

      This could break other things, so better may be to
      make a normalized copy of an expression. This could
      include simplification but in the case of Scans make
      the scheme more generic.

      @see FileScan.__eq__
      """
      return ZeroaryOperator.__eq__(self, other) and \
             self.relation_key == other.relation_key

    

def getTaggingFunc(t):
  """ 
  Return a visitor function that will tag 
  UnnamedAttributes with the provided TupleRef
  """

  def tagAttributes(expr):
    # TODO non mutable would be nice
    if isinstance(expr, expression.UnnamedAttributeRef):
      expr.tupleref = t

    return None
  
  return tagAttributes

class GrappaSymmetricHashJoin(algebra.Join, GrappaOperator):
    _i = 0

    @classmethod
    def __genHashName__(cls):
        name = "dhash_%03d" % cls._i;
        cls._i += 1
        return name

    def __genSyncName__(cls):
        name = "dh_sync_%03d" % cls._i;
        cls._i += 1
        return name

    def produce(self, state):
        if not isinstance(self.condition, expression.EQ):
            msg = "The C compiler can only handle equi-join conditions of a single attribute: %s" % self.condition
            raise ValueError(msg)

        init_template = """%(hashname)s.init_global_DHT( &%(hashname)s, cores()*5000 );
                        CompletionEvent %(syncname)s;
                        """
        spawn_start_template = """spawn(&%(syncname)s, [=] {
                        """
        finish_template = """});
        """
        wait_template = """%(syncname)s.wait();
        """

        self.outTuple = GrappaStagedTupleRef(gensym(), self.scheme())
        out_tuple_type_def = self.outTuple.generateDefinition()
        state.addDeclarations([out_tuple_type_def])

        # find the attribute that corresponds to the right child
        self.rightCondIsRightAttr = self.condition.right.position >= len(self.left.scheme())
        self.leftCondIsRightAttr = self.condition.left.position >= len(self.left.scheme())
        assert self.rightCondIsRightAttr^self.leftCondIsRightAttr

        self._hashname = self.__genHashName__()
        syncname = self.__genSyncName__()
        hashname = self._hashname

        self.right.childtag = "right"
        state.addCode(init_template%locals())
        # TODO: addCode is a hack: extract the synchronization code into pipeline plugins (like timing)
        state.addCode(spawn_start_template%locals())
        self.right.produce(state)
        state.addCode(finish_template)


        self.left.childtag = "left"
        state.addCode(spawn_start_template%locals())
        self.left.produce(state)
        state.addCode(finish_template)
        state.addCode(wait_template%locals())

    def consume(self, t, src, state):
        access_template = """%(hashname)s.insert_lookup_iter_%(side)s<&%(global_syncname)s>(%(keyname)s.get(%(keypos)s, %(keyname)s, [=](%(other_tuple_type)s %(valname)s) {
                                join_coarse_result_count++;
                                %(out_tuple_type)s %(out_tuple_name)s = combine<%(out_tuple_type)s,
                                                                                %(left_type)s,
                                                                                %(right_type)s> (%(left_name)s, %(right_name)s);
                                %(inner_plan_compiled)s
                                });
                                """
        global_sync_decl_template = """GlobalCompletionEvent %(global_syncname)s;
        """

        hashname = self._hashname
        keyname = t.name
        side = src.childtag

        outTuple = self.outTuple
        out_tuple_type = self.outTuple.getTupleTypename()
        out_tuple_name = self.outTuple.name

        global_syncname = gensym()
        state.addDeclarations([global_sync_decl_template%locals()])

        if src.childtag == "right":
            # save for later
            self.right_in_tuple_type = t.getTupleTypename()

            if self.rightCondIsRightAttr:
                keypos = self.condition.right.position-len(self.left.scheme())
            else:
                keypos = self.condition.left.position-len(self.left.scheme())


            inner_plan_compiled = self.parent.consume(outTuple, self, state)

            self.leftTypeRef = state.createUnresolvedSymbol()
            other_tuple_type = self.leftTypeRef.getPlaceholder()
            left_type = other_tuple_type
            right_type = self.right_in_tuple_type
            left_name = gensym()
            right_name = keyname
            self.right_name = right_name
            valname = left_name

            code = access_template % locals()
            return code

        if src.childtag == "left":
            declr_template =  """typedef DoubleDHT<int64_t, %(left_in_tuple_type)s, %(right_in_tuple_type)s, std_hash> DHT_%(left_in_tuple_type)s_%(right_in_tuple_type)s;
      DHT_%(left_in_tuple_type)s_%(right_in_tuple_type)s %(hashname)s;
      """
            right_in_tuple_type = self.right_in_tuple_type
            left_in_tuple_type = t.getTupleTypename()
            state.resolveSymbol(self.leftTypeRef, left_in_tuple_type)

            # declaration of hash map
            hashdeclr = declr_template % locals()
            state.addDeclarations([hashdeclr])

            if self.rightCondIsRightAttr:
                keypos = self.condition.left.position
            else:
                keypos = self.condition.right.position

            inner_plan_compiled = self.parent.consume(outTuple, self, state)

            left_type = left_in_tuple_type
            right_type = self.right_in_tuple_type
            other_tuple_type = self.right_in_tuple_type
            left_name = keyname
            right_name = gensym()
            valname = right_name

            code = access_template % locals()
            return code

        assert False, "src not equal to left or right"

class GrappaHashJoin(algebra.Join, GrappaOperator):
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
    
    self.right.childtag = "right"

    hashtableInfo = state.lookupExpr(self.right)
    if not hashtableInfo:
        # if right child never bound then store hashtable symbol and
        # call right child produce
        self._hashname = self.__genHashName__()
        LOG.debug("generate hashname %s for %s", self._hashname, self)

        init_template = """%(hashname)s.init_global_DHT( &%(hashname)s, cores()*5000 );"""
        setro_template = """%(hashname)s.set_RO_global( &%(hashname)s );"""
        hashname = self._hashname
        # surround the code for the pipeline with hash initialization and setRO
        state.addCode(init_template%locals())
        self.right.produce(state)
        state.addCode(setro_template%locals())
        state.saveExpr(self.right, (self._hashname, self.rightTupleTypename))
        #TODO always safe here? I really want to call saveExpr before self.right.produce(),
        #TODO but I need to get the self.rightTupleTypename cleanly
    else:
        # if found a common subexpression on right child then
        # use the same hashtable
        self._hashname, self.rightTupleTypename = hashtableInfo
        LOG.debug("reuse hash %s for %s", self._hashname, self)


    self.left.childtag = "left"
    self.left.produce(state)


  def consume(self, t, src, state):
    if src.childtag == "right":
      declr_template =  """typedef MatchesDHT<int64_t, %(in_tuple_type)s, std_hash> DHT_%(in_tuple_type)s;
      DHT_%(in_tuple_type)s %(hashname)s;
      """
      
      right_template = """%(hashname)s.insert(%(keyname)s.get(%(keypos)s), %(keyname)s);
      """   
      
      hashname = self._hashname
      keyname = t.name

      # find the attribute that corresponds to the right child
      rightCondIsRightAttr = self.condition.right.position >= len(self.left.scheme())
      leftCondIsRightAttr = self.condition.left.position >= len(self.left.scheme())
      assert rightCondIsRightAttr^leftCondIsRightAttr
      if rightCondIsRightAttr:
          keypos = self.condition.right.position-len(self.left.scheme())
      else:
          keypos = self.condition.left.position-len(self.left.scheme())


      in_tuple_type = t.getTupleTypename()
      self.rightTupleTypename = t.getTupleTypename()

      # declaration of hash map
      hashdeclr =  declr_template % locals()
      state.addDeclarations([hashdeclr])
      
      # materialization point
      code = right_template % locals()
      
      return code
    
    if src.childtag == "left":
      left_template = """
      %(hashname)s.lookup_iter( %(keyname)s.get(%(keypos)s), [=](%(right_tuple_type)s& %(right_tuple_name)s) {
        join_coarse_result_count++;
        %(out_tuple_type)s %(out_tuple_name)s = combine<%(out_tuple_type)s, %(keytype)s, %(right_tuple_type)s> (%(keyname)s, %(right_tuple_name)s);
        %(inner_plan_compiled)s
      });
    """

      hashname = self._hashname
      keyname = t.name
      keytype = t.getTupleTypename()
      keypos = self.condition.left.position
      
      right_tuple_name = gensym()
      right_tuple_type = self.rightTupleTypename

      outTuple = GrappaStagedTupleRef(gensym(), self.scheme())
      out_tuple_type_def = outTuple.generateDefinition()
      out_tuple_type = outTuple.getTupleTypename()
      out_tuple_name = outTuple.name

      state.addDeclarations([out_tuple_type_def])

      inner_plan_compiled = self.parent.consume(outTuple, self, state)
      
      code = left_template % locals()
      return code

    assert False, "src not equal to left or right"
      


def compile_filtering_join(joincondition, leftcondition, rightcondition):
    """return code for processing a filtering join"""


def indentby(code, level):
    indent = " " * ((level + 1) * 6)
    return "\n".join([indent + line for line in code.split("\n")])



#
#
#
#class FreeMemory(GrappaOperator):
#  def fire(self, expr):
#    for ref in noReferences(expr)


# Basic selection like serial C++      
class GrappaSelect(clangcommon.CSelect, GrappaOperator): pass

# Basic apply like serial C++
class GrappaApply(clangcommon.CApply, GrappaOperator): pass

# Basic duplication based bag union like serial C++
class GrappaUnionAll(clangcommon.CUnionAll, GrappaOperator): pass

# Basic materialized copy based project like serial C++
class GrappaProject(clangcommon.CProject, GrappaOperator): pass

class GrappaFileScan(clangcommon.CFileScan):
    ascii_scan_template_GRAPH = """
          {
            tuple_graph tg;
            tg = readTuples( "%(name)s" );

            FullEmpty<GlobalAddress<Graph<Vertex>>> f1;
            privateTask( [&f1,tg] {
              f1.writeXF( Graph<Vertex>::create(tg, /*directed=*/true) );
            });
            auto l_%(resultsym)s_index = f1.readFE();

            on_all_cores([=] {
              %(resultsym)s_index = l_%(resultsym)s_index;
            });
        }
        """

    # C++ type inference cannot infer T in readTuples<T>;
    # we resolve it later, so use %%
    ascii_scan_template = """
        {
        %(resultsym)s.data = readTuples<%%(result_type)s>( "%(name)s", FLAGS_nt);
        %(resultsym)s.numtuples = FLAGS_nt;
        auto l_%(resultsym)s = %(resultsym)s;
        on_all_cores([=]{ %(resultsym)s = l_%(resultsym)s; });
        }
        """

    def __get_ascii_scan_template__(self):
        return self.ascii_scan_template

    def __get_binary_scan_template__(self):
        LOG.warn("binary not currently supported for GrappaLanguage, emitting ascii")
        return self.ascii_scan_template



    
class swapJoinSides(rules.Rule):
  # swaps the inputs to a join
  def fire(self, expr):
    if isinstance(expr,algebra.Join):
      return algebra.Join(expr.condition, expr.right, expr.left)
    else:
      return expr

class GrappaAlgebra(object):
    language = GrappaLanguage

    operators = [
    #FileScan,
    MemoryScan,
    GrappaSelect,
    GrappaApply,
    GrappaProject,
    GrappaUnionAll,
    GrappaSymmetricHashJoin,
    GrappaHashJoin
  ]
    rules = [
  #  rules.removeProject(),
    rules.CrossProduct2Join(),
    #swapJoinSides(),
    rules.OneToOne(algebra.Select,GrappaSelect),
    rules.OneToOne(algebra.Apply, GrappaApply),
    rules.OneToOne(algebra.Scan,MemoryScan),
    #rules.OneToOne(algebra.Join,GrappaHashJoin),
    rules.OneToOne(algebra.Join,GrappaSymmetricHashJoin),
    rules.OneToOne(algebra.Project, GrappaProject),
    rules.OneToOne(algebra.Union,GrappaUnionAll) #TODO: obviously breaks semantics
  #  rules.FreeMemory()
  ]
