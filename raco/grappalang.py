
# TODO: To be refactored into parallel shared memory lang,
# where you plugin in the parallel shared memory language specific codegen

from raco import algebra
from raco import expression
from raco.language import Language
from raco import rules
from raco.pipelines import Pipelined
from raco.clangcommon import StagedTupleRef
from raco.clangcommon import ct
from raco import clangcommon

from algebra import gensym

import logging
LOG = logging.getLogger(__name__)

import os.path

template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "grappa_templates")


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
        return """LOG(INFO) << "%s";\n""" % txt

    @staticmethod
    def log_unquoted(code, level=0):
        if level == 0:
            log_str = "LOG(INFO)"
        else:
            log_str = "VLOG(%s)" % (level)

        return """%(log_str)s << %(code)s;\n""" % locals()

    @staticmethod
    def group_wrap(ident, grpcode, attrs):
        pipeline_template = ct("""
        Grappa::Metrics::reset();
        auto start_%(ident)s = walltime();
        %(grpcode)s
        auto end_%(ident)s = walltime();
        auto runtime_%(ident)s = end_%(ident)s - start_%(ident)s;
        %(timer_metric)s += runtime_%(ident)s;
        VLOG(1) << "pipeline %(ident)s: " << runtime_%(ident)s << " s";
        """)

        timer_metric = None
        if attrs['type'] == 'in_memory':
            timer_metric = "in_memory_runtime"
        elif attrs['type'] == 'scan':
            timer_metric = "saved_scan_runtime"

        code = pipeline_template % locals()
        return code

    @staticmethod
    def pipeline_wrap(ident, plcode, attrs):
        code = plcode

        syncname = attrs.get('sync')
        if syncname:
            inner_code = code
            sync_template = ct("""spawn(&%(syncname)s, [=] {
                    %(inner_code)s
                    });
                    """)
            code = sync_template % locals()

        syncname = attrs.get('syncdef')
        if syncname:
            inner_code = code
            sync_def_template = ct("""CompletionEvent %(syncname)s;
            %(inner_code)s
            """)
            code = sync_def_template % locals()

        return code

    @staticmethod
    def comment(txt):
        return "// %s\n" % txt

    nextstrid = 0

    @classmethod
    def newstringident(cls):
        r = """str_%s""" % (cls.nextstrid)
        cls.nextstrid += 1
        return r

    @classmethod
    def compile_numericliteral(cls, value):
        return '%s' % (value), [], []

    @classmethod
    def compile_stringliteral(cls, st):
        sid = cls.newstringident()
        decl = """int64_t %s;""" % (sid)
        init = """auto l_%(sid)s = string_index.string_lookup(%(st)s);
                   on_all_cores([=] { %(sid)s = l_%(sid)s; });""" % locals()
        return """(%s)""" % sid, [decl], [init]
        # raise ValueError("String Literals not supported in
        # C language: %s" % s)

    @classmethod
    def negation(cls, input):
        innerexpr, decls, inits = input
        return "(!%s)" % (innerexpr,), decls, inits

    @classmethod
    def expression_combine(cls, args, operator="&&"):
        opstr = " %s " % operator
        conjunc = opstr.join(["(%s)" % arg for arg, _, _ in args])
        decls = reduce(lambda sofar, x: sofar + x, [d for _, d, _ in args])
        inits = reduce(lambda sofar, x: sofar + x, [d for _, _, d in args])
        LOG.debug("conjunc: %s", conjunc)
        return "( %s )" % conjunc, decls, inits

    @classmethod
    def compile_attribute(cls, expr):
        if isinstance(expr, expression.NamedAttributeRef):
            raise TypeError("""Error compiling attribute reference %s. \
            C compiler only support unnamed perspective. \
            Use helper function unnamed.""" % expr)
        if isinstance(expr, expression.UnnamedAttributeRef):
            symbol = expr.tupleref.name
            # NOTE: this will only work in Selects right now
            position = expr.position
            return '%s.get(%s)' % (symbol, position), [], []


class GrappaOperator (Pipelined):
    language = GrappaLanguage

    def new_tuple_ref(self, sym, scheme):
        return GrappaStagedTupleRef(sym, scheme)


from algebra import UnaryOperator


class MemoryScan(algebra.UnaryOperator, GrappaOperator):
    def produce(self, state):
        self.input.produce(state)

    # TODO: when have pipeline tree representation,
    # will have a consumeMaterialized() method instead;
    # for now we reuse the tuple-based consume
    def consume(self, inputsym, src, state):
        # generate the materialization from file into memory

        # scan from index
        # memory_scan_template = """forall_localized( %(inputsym)s_index->vs, \
        # %(inputsym)s_index->nv, [](int64_t ai, Vertex& a) {
        #      forall_here_async<&impl::local_gce>( 0, a.nadj, \
        # [=](int64_t start, int64_t iters) {
        #      for (int64_t i=start; i<start+iters; i++) {
        #        auto %(tuple_name)s = a.local_adj[i];
        #
        #          %(inner_plan_compiled)s
        #       } // end scan over %(inputsym)s (for)
        #       }); // end scan over %(inputsym)s (forall_here_async)
        #       }); // end scan over %(inputsym)s (forall_localized)
        #       """

        global_sync_decl_template = ct("""
        GlobalCompletionEvent %(global_syncname)s;
        """)

        global_syncname = gensym()
        state.addDeclarations([global_sync_decl_template % locals()])
        state.setPipelineProperty('global_syncname', global_syncname)

        memory_scan_template = ct("""
    forall<&%(global_syncname)s>( %(inputsym)s.data, %(inputsym)s.numtuples, \
    [=](int64_t i, %(tuple_type)s& %(tuple_name)s) {
    %(inner_plan_compiled)s
    }); // end  scan over %(inputsym)s
    """)

        stagedTuple = state.lookupTupleDef(inputsym)
        tuple_type = stagedTuple.getTupleTypename()
        tuple_name = stagedTuple.name

        inner_plan_compiled = self.parent.consume(stagedTuple, self, state)

        code = memory_scan_template % locals()
        state.setPipelineProperty('type', 'in_memory')
        state.addPipeline(code)
        return None

    def shortStr(self):
        return "%s" % (self.opname())

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
        return UnaryOperator.__eq__(self, other)


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
    wait_template = ct("""%(syncname)s.wait();
        """)

    @classmethod
    def __genHashName__(cls):
        name = "dhash_%03d" % cls._i
        cls._i += 1
        return name

    def __genSyncName__(cls):
        name = "dh_sync_%03d" % cls._i
        cls._i += 1
        return name

    def produce(self, state):
        self.syncnames = []

        if not isinstance(self.condition, expression.EQ):
            msg = "The C compiler can only handle equi-join conditions\
             of a single attribute: %s" % self.condition
            raise ValueError(msg)

        init_template = ct("""%(hashname)s.init_global_DHT( &%(hashname)s, \
        cores()*16*1024 );
                        """)
        declr_template = ct("""typedef DoubleDHT<int64_t, \
                                                   %(left_in_tuple_type)s, \
                                                   %(right_in_tuple_type)s,
                                                std_hash> \
                    DHT_%(left_in_tuple_type)s_%(right_in_tuple_type)s;
      DHT_%(left_in_tuple_type)s_%(right_in_tuple_type)s %(hashname)s;
      """)
        # declaration of hash map
        self._hashname = self.__genHashName__()
        hashname = self._hashname
        self.leftTypeRef = state.createUnresolvedSymbol()
        left_in_tuple_type = self.leftTypeRef.getPlaceholder()
        self.rightTypeRef = state.createUnresolvedSymbol()
        right_in_tuple_type = self.rightTypeRef.getPlaceholder()
        hashdeclr = declr_template % locals()

        state.addDeclarationsUnresolved([hashdeclr])

        self.outTuple = GrappaStagedTupleRef(gensym(), self.scheme())
        out_tuple_type_def = self.outTuple.generateDefinition()
        state.addDeclarations([out_tuple_type_def])

        # find the attribute that corresponds to the right child
        self.rightCondIsRightAttr = \
            self.condition.right.position >= len(self.left.scheme())
        self.leftCondIsRightAttr = \
            self.condition.left.position >= len(self.left.scheme())
        assert self.rightCondIsRightAttr ^ self.leftCondIsRightAttr

        self.right.childtag = "right"
        state.addInitializers([init_template % locals()])
        self.right.produce(state)

        self.left.childtag = "left"
        self.left.produce(state)

        for sn in self.syncnames:
            syncname = sn
            state.addCode(self.wait_template % locals())

    def consume(self, t, src, state):
        access_template = ct("""
        %(hashname)s.insert_lookup_iter_%(side)s<&%(global_syncname)s>(\
        %(keyname)s.get(%(keypos)s), %(keyname)s, \
        [=](%(other_tuple_type)s %(valname)s) {
            join_coarse_result_count++;
            %(out_tuple_type)s %(out_tuple_name)s = \
                             combine<%(out_tuple_type)s, \
                                      %(left_type)s, \
                                      %(right_type)s> (%(left_name)s, \
                            %(right_name)s);
                                %(inner_plan_compiled)s
                                });
                                """)

        hashname = self._hashname
        keyname = t.name
        side = src.childtag

        outTuple = self.outTuple
        out_tuple_type = self.outTuple.getTupleTypename()
        out_tuple_name = self.outTuple.name

        syncname = self.__genSyncName__()
        state.setPipelineProperty('sync', syncname)
        state.setPipelineProperty('syncdef', syncname)
        self.syncnames.append(syncname)

        global_syncname = state.getPipelineProperty('global_syncname')

        if src.childtag == "right":

            # save for later
            self.right_in_tuple_type = t.getTupleTypename()
            state.resolveSymbol(self.rightTypeRef, self.right_in_tuple_type)

            if self.rightCondIsRightAttr:
                keypos = self.condition.right.position \
                    - len(self.left.scheme())
            else:
                keypos = self.condition.left.position \
                    - len(self.left.scheme())

            inner_plan_compiled = self.parent.consume(outTuple, self, state)

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
            right_in_tuple_type = self.right_in_tuple_type
            left_in_tuple_type = t.getTupleTypename()
            state.resolveSymbol(self.leftTypeRef, left_in_tuple_type)

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
        name = "hash_%03d" % cls._i
        cls._i += 1
        return name

    def produce(self, state):
        if not isinstance(self.condition, expression.EQ):
            msg = "The C compiler can only handle equi-join conditions of\
             a single attribute: %s" % self.condition
            raise ValueError(msg)

        declr_template = ct("""typedef MatchesDHT<int64_t, \
                          %(in_tuple_type)s, std_hash> \
                           DHT_%(in_tuple_type)s;
        DHT_%(in_tuple_type)s %(hashname)s;
        """)

        self.right.childtag = "right"
        self.rightTupleTypeRef = None  # may remain None if CSE succeeds

        # find the attribute that corresponds to the right child
        self.rightCondIsRightAttr = \
            self.condition.right.position >= len(self.left.scheme())
        self.leftCondIsRightAttr = \
            self.condition.left.position >= len(self.left.scheme())
        assert self.rightCondIsRightAttr ^ self.leftCondIsRightAttr

        hashtableInfo = state.lookupExpr(self.right)
        if not hashtableInfo:
            # if right child never bound then store hashtable symbol and
            # call right child produce
            self._hashname = self.__genHashName__()
            LOG.debug("generate hashname %s for %s", self._hashname, self)

            hashname = self._hashname

            # declaration of hash map
            self.rightTupleTypeRef = state.createUnresolvedSymbol()
            in_tuple_type = self.rightTupleTypeRef.getPlaceholder()
            hashdeclr = declr_template % locals()
            state.addDeclarationsUnresolved([hashdeclr])

            init_template = ct("""%(hashname)s.init_global_DHT( &%(hashname)s,
            cores()*16*1024 );""")
            state.addInitializers([init_template % locals()])
            self.right.produce(state)
            state.saveExpr(self.right,
                           (self._hashname, self.rightTupleTypename))
            # TODO always safe here? I really want to call
            # TODO saveExpr before self.right.produce(),
            # TODO but I need to get the self.rightTupleTypename cleanly
        else:
            # if found a common subexpression on right child then
            # use the same hashtable
            self._hashname, self.rightTupleTypename = hashtableInfo
            LOG.debug("reuse hash %s for %s", self._hashname, self)

        self.left.childtag = "left"
        self.left.produce(state)

    def consume(self, t, src, state):
        if src.childtag == "right":

            right_template = ct("""
            %(hashname)s.insert(%(keyname)s.get(%(keypos)s), %(keyname)s);
            """)

            hashname = self._hashname
            keyname = t.name

            if self.rightCondIsRightAttr:
                keypos = self.condition.right.position \
                    - len(self.left.scheme())
            else:
                keypos = self.condition.left.position \
                    - len(self.left.scheme())

            self.rightTupleTypename = t.getTupleTypename()
            if self.rightTupleTypeRef is not None:
                state.resolveSymbol(self.rightTupleTypeRef,
                                    self.rightTupleTypename)

            # materialization point
            code = right_template % locals()

            return code

        if src.childtag == "left":
            left_template = ct("""
            %(hashname)s.lookup_iter<&%(pipeline_sync)s>( \
            %(keyname)s.get(%(keypos)s), \
            [=](%(right_tuple_type)s& %(right_tuple_name)s) {
              join_coarse_result_count++;
              %(out_tuple_type)s %(out_tuple_name)s = \
               combine<%(out_tuple_type)s, \
                       %(keytype)s, \
                       %(right_tuple_type)s> \
                           (%(keyname)s, %(right_tuple_name)s);
              %(inner_plan_compiled)s
            });
     """)

            hashname = self._hashname
            keyname = t.name
            keytype = t.getTupleTypename()

            pipeline_sync = state.getPipelineProperty('global_syncname')

            if self.rightCondIsRightAttr:
                keypos = self.condition.left.position
            else:
                keypos = self.condition.right.position

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


def indentby(code, level):
    indent = " " * ((level + 1) * 6)
    return "\n".join([indent + line for line in code.split("\n")])

#
#
#
# class FreeMemory(GrappaOperator):
#  def fire(self, expr):
#    for ref in noReferences(expr)


# Basic selection like serial C++
class GrappaSelect(clangcommon.CSelect, GrappaOperator):
    pass


# Basic apply like serial C++
class GrappaApply(clangcommon.CApply, GrappaOperator):
    pass


# Basic duplication based bag union like serial C++
class GrappaUnionAll(clangcommon.CUnionAll, GrappaOperator):
    pass


# Basic materialized copy based project like serial C++
class GrappaProject(clangcommon.CProject, GrappaOperator):
    pass


class GrappaFileScan(clangcommon.CFileScan, GrappaOperator):
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
    if (FLAGS_bin) {
    %(resultsym)s = readTuplesUnordered<%%(result_type)s>( "%(name)s.bin" );
    } else {
    %(resultsym)s.data = readTuples<%%(result_type)s>( "%(name)s", FLAGS_nt);
    %(resultsym)s.numtuples = FLAGS_nt;
    auto l_%(resultsym)s = %(resultsym)s;
    on_all_cores([=]{ %(resultsym)s = l_%(resultsym)s; });
    }
    }
    """

    def __get_ascii_scan_template__(self):
        return self.ascii_scan_template

    def __get_binary_scan_template__(self):
        LOG.warn("binary not currently supported\
         for GrappaLanguage, emitting ascii")
        return self.ascii_scan_template

    def __get_relation_decl_template__(self):
        return """Relation<%(tuple_type)s> %(resultsym)s;"""


class MemoryScanOfFileScan(rules.Rule):
    """A rewrite rule for making a scan into materialization
     in memory then memory scan"""
    def fire(self, expr):
        if isinstance(expr, algebra.Scan) \
                and not isinstance(expr, GrappaFileScan):
            return MemoryScan(GrappaFileScan(expr.relation_key, expr.scheme()))
        return expr

    def __str__(self):
        return "Scan => MemoryScan(FileScan)"


class swapJoinSides(rules.Rule):
    # swaps the inputs to a join
    def fire(self, expr):
        if isinstance(expr, algebra.Join):
            return algebra.Join(expr.condition, expr.right, expr.left)
        else:
            return expr


class GrappaAlgebra(object):
    language = GrappaLanguage

    operators = [
        # FileScan,
        MemoryScan,
        GrappaSelect,
        GrappaApply,
        GrappaProject,
        GrappaUnionAll,
        GrappaSymmetricHashJoin,
        GrappaHashJoin
    ]

    rules = [
        # rules.removeProject(),
        rules.CrossProduct2Join(),
        # swapJoinSides(),
        rules.OneToOne(algebra.Select, GrappaSelect),
        rules.OneToOne(algebra.Apply, GrappaApply),
        # rules.OneToOne(algebra.Scan,MemoryScan),
        MemoryScanOfFileScan(),
        #  rules.OneToOne(algebra.Join, GrappaSymmetricHashJoin),
        rules.OneToOne(algebra.Join, GrappaHashJoin),
        rules.OneToOne(algebra.Project, GrappaProject),
        # TODO: this Union obviously breaks semantics
        rules.OneToOne(algebra.Union, GrappaUnionAll)
        # rules.FreeMemory()
    ]
