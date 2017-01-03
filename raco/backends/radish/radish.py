
# TODO: To be refactored into parallel shared memory lang,
# where you plugin in the parallel shared memory language specific codegen

import logging

import raco.scheme as scheme
from raco import algebra
from raco import expression
from raco import rules
from raco.algebra import gensym
from raco.backends import Algebra
from raco.backends.cpp import cppcommon
from raco.backends.cpp.cppcommon import StagedTupleRef, CBaseLanguage
from raco.expression import aggregate
from raco.pipelines import Pipelined
from raco.utility import emitlist

_LOG = logging.getLogger(__name__)

import itertools


class _ARRAY_REPRESENTATION:
    GLOBAL_ARRAY = 'global_array'
    SYMMETRIC_ARRAY = 'symmetric_array'


def define_cl_arg(type, name, default_value, description):
    return GrappaLanguage.cgenv().get_template(
        'define_cl_arg.cpp').render(locals())


class GrappaStagedTupleRef(StagedTupleRef):

    def __init__(self, relsym, scheme, aligned=False):
        self.aligned = aligned
        super(GrappaStagedTupleRef, self).__init__(relsym, scheme)

    def __afterDefinitionCode__(self, numfields, fieldtypes):
        # Grappa requires structures to be block aligned if they will be
        # iterated over with localizing forall
        if self.aligned:
            return "GRAPPA_BLOCK_ALIGNED"
        else:
            return ""


class GrappaLanguage(CBaseLanguage):
    _template_path = 'radish/grappa_templates'
    _cgenv = CBaseLanguage.__get_env_for_template_libraries__(_template_path)

    @classmethod
    def on_all(cls, code):
        return """on_all_cores([=] {{
        {code}
        }});
        """.format(code=code)

    @classmethod
    def cgenv(cls):
        return cls._cgenv

    @classmethod
    def base_template(cls):
        return cls.cgenv().get_template('base_query.cpp')

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

    @classmethod
    def compile_stringliteral(cls, st):
        if cls._external_indexing:
            st = cls.c_stringify(st)
            sid = cls.newstringident()
            decl = """int64_t %s;""" % (sid)
            lookup_init = GrappaLanguage.cgenv().get_template(
                'string_index_lookup.cpp').render(locals())
            build_init = """
            string_index = build_string_index("sp2bench.index");
            """

            return """(%s)""" % sid, [decl], [build_init, lookup_init]
            # raise ValueError("String Literals not supported in
            # C language: %s" % s)
        else:
            return super(GrappaLanguage, cls).compile_stringliteral(st)

    @staticmethod
    def group_wrap(ident, grpcode, attrs):
        timing_template = GrappaLanguage.cgenv().get_template(
            'grappa_group_timing.cpp')
        inner_code = grpcode

        timer_metric = None
        if attrs['type'] == 'in_memory':
            timer_metric = "in_memory_runtime"
            # only trace in_memory
            tracing_on = "Grappa::Metrics::start_tracing();"
            tracing_off = "Grappa::Metrics::stop_tracing();"
        elif attrs['type'] == 'scan':
            timer_metric = "saved_scan_runtime"
            tracing_on = ""
            tracing_off = ""

        code = emitlist(["Grappa::Metrics::reset();",
                         timing_template.render(locals())])

        return code

    @staticmethod
    def iterators_wrap(code, attrs):
        code = GrappaLanguage.on_all(code)
        code += """
        iterate(&{fragment_symbol}, &{global_syncname});
        """.format(**attrs)
        return code

    @staticmethod
    def pipeline_wrap(ident, plcode, attrs):

        def apply_wrappers(code, wrappers):
            """
            Wraps the code successively with wrappers.
            First wrapper is innermost

            @param code the initial code to wrap
            @param wrappers tuple of format (template, bindings).
            The template must include {{inner_code}}
            """
            current_result = code
            for template, bindings in wrappers:
                allbindings = bindings.copy()
                allbindings.update({'inner_code': current_result})
                current_result = template.render(allbindings)

            return current_result

        wrappers = []

        dependences = attrs.get('dependences', set())
        assert isinstance(dependences, set)
        _LOG.debug("pipeline %s dependences %s", ident, dependences)

        name = "pipeline_{ident}".format(ident=ident)

        syncname = attrs.get('sync')
        if syncname:
            dependence_captures = emitlist(
                ["&{dep},".format(dep=d) for d in dependences])
            sync_template = GrappaLanguage.cgenv().get_template('spawn.cpp')
            wrappers.append((sync_template, locals()))

        return apply_wrappers(plcode, wrappers)


class GrappaOperator (Pipelined, algebra.Operator):
    _language = GrappaLanguage

    @classmethod
    def new_tuple_ref(cls, sym, scheme):
        return GrappaStagedTupleRef(sym, scheme)

    @classmethod
    def language(cls):
        return cls._language

    def postorder_traversal(self, func):
        return self.postorder(func)


from raco.algebra import UnaryOperator


def create_pipeline_synchronization(state):
    """
    The pipeline_synchronization will sync tasks
    within a single pipeline. Adds this new object to
    the compiler state.
    """
    global_syncname = gensym()

    # true = tracked by gce user metrics

    global_sync_decl = GrappaLanguage.cgenv().get_template(
        'sync_declaration.cpp').render(locals())

    gce_metric_template = GrappaLanguage.cgenv().get_template(
        'gce_app_metric.cpp')

    pipeline_id = state.getCurrentPipelineId()
    gce_metric_def = gce_metric_template.render(locals())

    state.addDeclarations([global_sync_decl, gce_metric_def])

    state.setPipelineProperty('global_syncname', global_syncname)
    return global_syncname


class GrappaSingletonRelation(algebra.SingletonRelation, GrappaOperator):

    def produce(self, state):
        stagedTuple = self.new_tuple_ref(gensym(), self.scheme())

        state.addDeclarations([stagedTuple.generateDefinition(),
                               "{type} {name};".format(
                                   type=stagedTuple.getTupleTypename(),
                                   name=stagedTuple.name
        )])
        inner_code = self.parent().consume(stagedTuple, self, state)

        get_pipeline_task_name(state)
        state.setPipelineProperty('type', 'in_memory')
        state.setPipelineProperty('source', self.__class__)
        state.addPipeline(inner_code)

    def consume(self, t, src, state):
        assert False, "This is a source; it does not consume"

    def shortStr(self):
        return "GrappaSingletonRelation"


# TODO: make filescan pipeline turn into
# TODO sequence(storetemp(filescan)), scantemp).
# TODO The MemoryScan should just already have the name
class GrappaMemoryScan(algebra.UnaryOperator, GrappaOperator):

    def __init__(self, inp,
                 representation=_ARRAY_REPRESENTATION.GLOBAL_ARRAY,
                 symbol_name=None,
                 analyzed_num_tuples=10000):
        self.array_representation = representation
        self.symbol_name = symbol_name
        self.analyzed_num_tuples = analyzed_num_tuples
        super(GrappaMemoryScan, self).__init__(inp)

    def num_tuples(self):
        return self.analyzed_num_tuples

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

        global_syncname = create_pipeline_synchronization(state)
        get_pipeline_task_name(state)

        # check if the input symbol was from a temp relation
        if isinstance(inputsym, TempRelationSymbol):
            inputsym_raw = inputsym.symbol()
            inputsym = inputsym.symbol()
            readfrom = "{inputsym}.read()".format(inputsym=inputsym)
        else:
            inputsym_raw = inputsym
            readfrom = "{inputsym}.data".format(inputsym=inputsym)

        # necessary to look this up rather than create a new
        # one because we need to read from the correct type provided
        # by the File/Temp relation input
        stagedTuple = state.lookupTupleDef(inputsym_raw)

        # get template for the scan/iteration
        memory_scan_template_name = {
            _ARRAY_REPRESENTATION.GLOBAL_ARRAY:
            'global_array_memory_scan.cpp',
            _ARRAY_REPRESENTATION.SYMMETRIC_ARRAY:
            'symmetric_array_memory_scan.cpp'
        }[self.array_representation]
        memory_scan_template = self.language().cgenv().get_template(
            memory_scan_template_name)

        tuple_type = stagedTuple.getTupleTypename()
        tuple_name = stagedTuple.name

        inner_code = self.parent().consume(stagedTuple, self, state)

        code = memory_scan_template.render(locals())
        state.setPipelineProperty('type', 'in_memory')
        state.setPipelineProperty('source', self.__class__)
        state.addPipeline(code)
        return None

    def shortStr(self):
        return "{op}({rep}, {debug_info}) "\
            .format(op=self.opname(),
                    rep=self.array_representation,
                    debug_info=self.symbol_name)

    def partitioning(self):
        raise NotImplementedError()

    def __eq__(self, other):
        """
        See important __eq__ notes below
        @see FileScan.__eq__
        """
        return UnaryOperator.__eq__(self, other)

    def __repr__(self):
        return "{op}({inp!r}, {rep!r})".format(op=self.opname(),
                                               inp=self.input,
                                               rep=self.array_representation)


class CKeyUtils:

    @staticmethod
    def _aggregate_val(tuple, cols):
        return "std::make_tuple({0})".format(
            ','.join([tuple.get_code(p) for p in cols]))

    @staticmethod
    def _aggregate_type(language, sch, cols):
        return "std::tuple<{0}>".format(
            ','.join([language.typename(
                expression.UnnamedAttributeRef(c).typeof(sch, None))
                for c in cols]))


class GrappaJoin(algebra.Join, GrappaOperator):
    pass


class GrappaSymmetricHashJoin(GrappaJoin, GrappaOperator):
    _i = 0

    @classmethod
    def __genBaseName__(cls):
        name = "%03d" % cls._i
        cls._i += 1
        return name

    def __getHashName__(self):
        name = "%s_dhash_%s" % (self.__class__.__name__, self.symBase)
        return name

    def __init__(self, *args):
        super(GrappaSymmetricHashJoin, self).__init__(*args)
        self._cgenv = cppcommon.prepend_template_relpath(
            self.language().cgenv(),
            '{0}/symmetrichashjoin'.format(GrappaLanguage._template_path))

    def _force_right_to_left_sync(self):
        """Returns whether or not to create a dependence from right
        pipeline to left pipeline. Default symmetric hash join plan allows
        the two sides to be concurrent"""
        return False

    def produce(self, state):
        self.symBase = self.__genBaseName__()

        init_template = self._cgenv.get_template('hash_init.cpp')

        declr_template = self._cgenv.get_template('hash_declaration.cpp')

        my_sch = self.scheme()
        left_sch = self.left.scheme()
        right_sch = self.right.scheme()

        self.leftcols, self.rightcols = \
            algebra.convertcondition(self.condition,
                                     len(left_sch),
                                     left_sch + right_sch)

        # declaration of hash map
        self._hashname = self.__getHashName__()
        keytype = CKeyUtils._aggregate_type(
            self.language(),
            right_sch,
            self.rightcols)
        hashname = self._hashname
        self.leftTypeRef = state.createUnresolvedSymbol()
        left_in_tuple_type = self.leftTypeRef.getPlaceholder()
        self.rightTypeRef = state.createUnresolvedSymbol()
        right_in_tuple_type = self.rightTypeRef.getPlaceholder()
        hashdeclr = declr_template.render(locals())

        state.addDeclarationsUnresolved([hashdeclr])

        self.outTuple = GrappaStagedTupleRef(gensym(), my_sch)
        out_tuple_type_def = self.outTuple.generateDefinition()
        state.addDeclarations([out_tuple_type_def])

        self.right.childtag = "right"
        self.initializer = init_template.render(locals())
        state.addInitializers([self.initializer])
        self.right.produce(state)

        self.left.childtag = "left"
        self.left.produce(state)

    def consume(self, t, src, state):
        access_template = self._cgenv.get_template('hash_insert_lookup.cpp')

        hashname = self._hashname
        keyname = t.name
        side = src.childtag

        outTuple = self.outTuple
        out_tuple_type = self.outTuple.getTupleTypename()
        out_tuple_name = self.outTuple.name

        global_syncname = state.getPipelineProperty('global_syncname')

        # note that this recycles both sides of the hash join structure
        state.recordCodeWhenInLoop(
            self.language().comment("recycle") +
            "{hashname}.clear();\n".format(
                hashname=hashname))

        if src.childtag == "right":
            left_sch = self.left.scheme()

            # save for later
            self.right_in_tuple_type = t.getTupleTypename()
            state.resolveSymbol(self.rightTypeRef, self.right_in_tuple_type)

            inner_plan_compiled = self.parent().consume(outTuple, self, state)

            keyval = CKeyUtils._aggregate_val(t, self.rightcols)

            other_tuple_type = self.leftTypeRef.getPlaceholder()
            left_type = other_tuple_type
            right_type = self.right_in_tuple_type
            left_name = gensym()
            right_name = keyname
            self.right_name = right_name
            valname = left_name

            append_func_name, combine_function_def = \
                GrappaStagedTupleRef.get_append(
                    out_tuple_type,
                    left_type, len(left_sch),
                    right_type, len(t.scheme))

            # Need to add later because requires left tuple type decl.
            # Needs to be a list because could be multiple right sides
            # occurences
            if not hasattr(self, 'right_combine_decls'):
                self.right_combine_decls = []
            self.right_combine_decls.append(combine_function_def)

            if self._force_right_to_left_sync():
                self.right_syncname = get_pipeline_task_name(state)

            code = access_template.render(locals())
            return code

        if src.childtag == "left":
            if self._force_right_to_left_sync():
                state.addToPipelinePropertySet(
                    'dependences',
                    self.right_syncname)

            right_in_tuple_type = self.right_in_tuple_type
            left_in_tuple_type = t.getTupleTypename()
            state.resolveSymbol(self.leftTypeRef, left_in_tuple_type)

            keyval = CKeyUtils._aggregate_val(t, self.leftcols)

            inner_plan_compiled = self.parent().consume(outTuple, self, state)

            left_type = left_in_tuple_type
            right_type = self.right_in_tuple_type
            other_tuple_type = self.right_in_tuple_type
            left_name = keyname
            right_name = gensym()
            valname = right_name

            append_func_name, combine_function_def = \
                GrappaStagedTupleRef.get_append(
                    out_tuple_type,
                    left_type, len(t.scheme),
                    right_type, len(self.right.scheme()))

            state.addDeclarations(
                self.right_combine_decls +
                [combine_function_def])

            code = access_template.render(locals())
            return code

        assert False, "src not equal to left or right"


class GrappaOverSynchronizedSymmetricHashJoin(GrappaSymmetricHashJoin):

    def _force_right_to_left_sync(self):
        return True


class GrappaShuffleHashJoin(algebra.Join, GrappaOperator):
    # this class is out-of-date, but can be fixed
    pass
#    _i = 0
#
#    @classmethod
#    def __genBaseName__(cls):
#        name = "%03d" % cls._i
#        cls._i += 1
#        return name
#
#    def __getHashName__(self):
#        name = "hashjoin_reducer_%s" % self.symBase
#        return name
#
#    def __init__(self, *args):
#        super(GrappaShuffleHashJoin, self).__init__(*args)
#        self._cgenv = cppcommon.prepend_template_relpath(
#            self.language().cgenv(),
#            '{0}/shufflehashjoin'.format(GrappaLanguage._template_path))
#
#    def produce(self, state):
#        left_sch = self.left.scheme()
#
#        self.syncnames = []
#        self.symBase = self.__genBaseName__()
#
#        self.right.childtag = "right"
#        self.rightTupleTypeRef = None  # may remain None if CSE succeeds
#        self.leftTupleTypeRef = None  # may remain None if CSE succeeds
#
#        # find the attribute that corresponds to the right child
#        self.rightCondIsRightAttr = \
#            self.condition.right.position >= len(left_sch)
#        self.leftCondIsRightAttr = \
#            self.condition.left.position >= len(left_sch)
#        assert self.rightCondIsRightAttr ^ self.leftCondIsRightAttr
#
#        # find right key position
#        if self.rightCondIsRightAttr:
#            self.right_keypos = self.condition.right.position \
#                - len(left_sch)
#        else:
#            self.right_keypos = self.condition.left.position \
#                - len(left_sch)
#
#        # find left key position
#        if self.rightCondIsRightAttr:
#            self.left_keypos = self.condition.left.position
#        else:
#            self.left_keypos = self.condition.right.position
#
#        # define output tuple
#        outTuple = GrappaStagedTupleRef(gensym(), self.scheme())
#        out_tuple_type_def = outTuple.generateDefinition()
#        out_tuple_type = outTuple.getTupleTypename()
#        out_tuple_name = outTuple.name
#
#        # common index is defined by same right side and same key
#        # TODO: probably want also left side
#        hashtableInfo = state.lookupExpr((self.right, self.right_keypos))
#        if not hashtableInfo:
#            # if right child never bound then store hashtable symbol and
#            # call right child produce
#            self._hashname = self.__getHashName__()
#            _LOG.debug("generate hashname %s for %s", self._hashname, self)
#
#            hashname = self._hashname
#
#            # declaration of hash map
#            self.rightTupleTypeRef = state.createUnresolvedSymbol()
#            self.leftTupleTypeRef = state.createUnresolvedSymbol()
#            self.outTupleTypeRef = state.createUnresolvedSymbol()
#            right_type = self.rightTupleTypeRef.getPlaceholder()
#            left_type = self.leftTupleTypeRef.getPlaceholder()
#
#            # TODO: really want this addInitializers to be addPreCode
#            # TODO: *for all pipelines that use this hashname*
#            init_template = self._cgenv.get_template('hash_init.cpp')
#
#            state.addInitializers([init_template.render(locals())])
#            self.right.produce(state)
#
#            self.left.childtag = "left"
#            self.left.produce(state)
#
#            state.saveExpr((self.right, self.right_keypos),
#                           (self._hashname, right_type, left_type,
#                            self.right_syncname, self.left_syncname))
#
#        else:
#            # if found a common subexpression on right child then
#            # use the same hashtable
#            self._hashname, right_type, left_type,\
#                self.right_syncname, self.left_syncname = hashtableInfo
#            _LOG.debug("reuse hash %s for %s", self._hashname, self)
#
#        # now that Relation is produced,
#        # produce its contents by iterating over
#        # the join result
#        iterate_template = self._cgenv.get_template('result_scan.cpp')
#
#        hashname = self._hashname
#
#        state.addDeclarations([out_tuple_type_def])
#
#        pipeline_sync = create_pipeline_synchronization(state)
#        get_pipeline_task_name(state)
#
#        # add dependences on left and right inputs
#        state.addToPipelinePropertySet('dependences', self.right_syncname)
#        state.addToPipelinePropertySet('dependences', self.left_syncname)
#
#        # reduce is a single self contained pipeline.
#        # future hashjoin implementations may pipeline out of it
#        # by passing a continuation to reduceExecute
#        reduce_template = self._cgenv.get_template('reduce.cpp')
#
#        state.addPreCode(reduce_template.render(locals()))
#
#        delete_template = self._cgenv.get_template('delete.cpp')
#
#        state.addPostCode(delete_template.render(locals()))
#
#        inner_code_compiled = self.parent().consume(outTuple, self, state)
#
#        code = iterate_template % locals()
#        state.setPipelineProperty('type', 'in_memory')
#        state.setPipelineProperty('source', self.__class__)
#        state.addPipeline(code)

#    def consume(self, inputTuple, fromOp, state):
#        if fromOp.childtag == "right":
#            side = "Right"
#            self.right_syncname = get_pipeline_task_name(state)
#
#            keypos = self.right_keypos
#
#            self.rightTupleTypename = inputTuple.getTupleTypename()
#            if self.rightTupleTypeRef is not None:
#                state.resolveSymbol(self.rightTupleTypeRef,
#                                    self.rightTupleTypename)
#        elif fromOp.childtag == "left":
#            side = "Left"
#            self.left_syncname = get_pipeline_task_name(state)
#
#            keypos = self.left_keypos
#
#            self.leftTupleTypename = inputTuple.getTupleTypename()
#            if self.leftTupleTypeRef is not None:
#                state.resolveSymbol(self.leftTupleTypeRef,
#                                    self.leftTupleTypename)
#        else:
#            assert False, "src not equal to left or right"
#
#        hashname = self._hashname
#        keyname = inputTuple.name
#        keytype = inputTuple.getTupleTypename()
#        keyval = inputTuple.get_code(keypos)
#
#        # intra-pipeline sync
#        global_syncname = state.getPipelineProperty('global_syncname')
#
#        mat_template = self._cgenv.get_template('materialize.cpp')
#
#        # materialization point
#        code = mat_template.render(locals())
#        return code


class GrappaGroupBy(cppcommon.BaseCGroupby, GrappaOperator):
    _i = 0

    @classmethod
    def __genHashName__(cls):
        name = "%s_hash_%03d" % (cls.__name__, cls._i)
        cls._i += 1
        return name

    def __init__(self, *args):
        super(GrappaGroupBy, self).__init__(*args)
        self._cgenv = cppcommon.prepend_template_relpath(
            self.language().cgenv(),
            '{0}/groupby'.format(GrappaLanguage._template_path))

    def _combiner_for_builtin_update(self, update_op):
        # FIXME: should be using get_decomposable_state instead of this hack
        # FIXME: need AVG and STDEV
        if update_op.__class__ == aggregate.COUNT \
                or update_op.__class__ == aggregate.COUNTALL:
            return aggregate.SUM(update_op.input)
        else:
            return update_op

    def _init_func_for_op(self, op):
        r = {
            aggregate.MAX: 'std::numeric_limits<{state_type}>::lowest',
            aggregate.MIN: 'std::numeric_limits<{state_type}>::max'
        }.get(op.__class__)
        if r is None:
            return 'Aggregates::Zero<{state_type}>'
        else:
            return r

    def _reconstruct_aggregates_schema(self):
        """
         reconstruct the lost mapping of schema to aggregate
         expressions/grouping list
        """

        # TODO: doesn't this exist somewhere in raco?
        def resolve_name(ref, sch):
            if isinstance(
                    ref,
                    expression.UnnamedAttributeRef)or isinstance(
                    ref,
                    expression.UnnamedAttributeRef):
                return sch.get_names()[ref.position]
            else:
                return ref.name

        inp_sch = self.input.scheme()

        grouped_names = set([resolve_name(ref, inp_sch)
                             for ref in self.grouping_list])
        aggregates_types = [typ  # throw away the name because it is made up
                            for name, typ in self.scheme()
                            if name not in grouped_names]
        aggregates_names = [
            resolve_name(
                a.input,
                inp_sch) for a in self.aggregate_list]

        return scheme.Scheme(zip(aggregates_names, aggregates_types))

    def _assignment_code(self, output_tuple):
        """
         assign state type tuple to output type tuple
         For 0key case they should just be treated the same,
         i.e., self.state_tuple same as output_tuple for !self.useKey
         except for name
         """
        assignmentcode = ""
        for i in range(0, len(output_tuple.scheme)):
            d = output_tuple.set_func_code(i)
            s = output_tuple.get_code_with_name(
                i, "{0}_tmp".format(
                    output_tuple.name))
            assignment_template = self._cgenv.get_template('assignment.cpp')
            assignmentcode += assignment_template.render(
                dst_set_func=d, src_expr_compiled=s)

        return assignmentcode

    def _impl_produce(self, state):
        pipeline_sync = create_pipeline_synchronization(state)
        get_pipeline_task_name(state)

        if self.useKey:
            inp_sch = self.input.scheme()

            numkeys = len(self.grouping_list)
            keytype = self._key_type(inp_sch)
            mapping_var_name = gensym()
            emit_type = self.state_tuple.getTupleTypename()

            # pass in attribute values individually
            produce_template = self._cgenv.get_template(
                'multi_uda_scan.cpp')
        else:
            produce_template = self._cgenv.get_template(
                'multi_uda_0key_output.cpp')

        output_tuple = GrappaStagedTupleRef(gensym(), self.scheme())
        output_tuple_name = output_tuple.name
        output_tuple_type = output_tuple.getTupleTypename()
        output_tuple_set_func = output_tuple.set_func_code(0)  # UNUSED??
        state.addDeclarations([output_tuple.generateDefinition()])

        inner_code = self.parent().consume(output_tuple, self, state)
        comment = self.language().comment("scan of " + str(self))

        assignmentcode = self._assignment_code(output_tuple)

        hashname = self._hashname
        state_type = self.state_tuple.getTupleTypename()
        combine_func = self.combine_func

        code = produce_template.render(locals())
        state.setPipelineProperty("type", "in_memory")
        state.addPipeline(code)

    def _key_type(self, inp_sch):
        return "std::tuple<{types}>".format(
            types=','.join([self.language().typename(
                g.typeof(inp_sch, None)) for g in self.grouping_list]))

    def _init_template(self):
        return self._cgenv.get_template('withkey_init.cpp')

    def _reuse_properties(self, other):
        self._hashname = other._hashname
        self.func_name = other.func_name
        self.state_tuple = other.state_tuple
        self.input_syncname = other.input_syncname
        self.input_type_ref = other.input_type_ref

    def produce(self, state):
        # we distinguish between no-key and using a key cases
        self.useKey = len(self.grouping_list) > 0
        _LOG.debug("groupby uses keys? %s" % self.useKey)

        # restrictions on aggregate support
        if not(self.useKey or
               all([not isinstance(exp, expression.UdaAggregateExpression)
                    for exp in self.aggregate_list])):
            raise NotImplementedError("""
                    UDAs with no groupby key. The reason is that we
                    need to support decomposable state for correctness
                    of a local aggregate and global combine strategy. This
                    is solved adhoc in specific important builtin cases
                    like COUNT""")

        if not all([not isinstance(exp, expression.ZeroaryOperator)
                    for exp in self.aggregate_list]):
            raise NotImplementedError("""No support for Zeroary aggregates yet.
            If using COUNT(*), then use COUNT(a), but NOTE that COUNT(a)
            does not have proper null semantics
            (unconditionally counts everything)""")

        self.aggregates_schema = self._reconstruct_aggregates_schema()

        hashtableInfo = state.lookupExpr(self)
        if hashtableInfo:
            subexpression_proxy = hashtableInfo
            self._reuse_properties(subexpression_proxy)
        else:
            symbol = gensym()
            self._hashname = GrappaGroupBy.__genHashName__()
            _LOG.debug("generate hashname %s for %s", self._hashname, self)
            self.func_name = "__{0}".format(symbol)

            self.state_tuple = GrappaStagedTupleRef(symbol,
                                                    self.aggregates_schema,
                                                    aligned=(not self.useKey))
            self.input_type_ref = state.createUnresolvedSymbol()
            state.saveExpr(self, self)
            state.addDeclarations([self.state_tuple.generateDefinition()])

        state_type = self.state_tuple.getTupleTypename()
        self.update_func = "{name}_update".format(name=self.func_name)
        self.init_func = "{name}_init".format(name=self.func_name)
        # combine_func currently just for 0key_output
        self.combine_func = "{name}_combine".format(name=self.func_name)
        init_func = self.init_func
        update_func = self.update_func
        combine_func = self.combine_func

        inp_sch = self.input.scheme()

        if self.useKey:
            numkeys = len(self.grouping_list)
            keytype = self._key_type(inp_sch)

        hashname = self._hashname

        if self.useKey:
            init_template = self._init_template()
            update_val_type = self.input_type_ref.getPlaceholder()
            valtype = state_type
            decl_template = self._cgenv.get_template('withkey_decl.cpp')
            state.addDeclarations([decl_template.render(locals())])
        else:
            no_key_state_initializer = \
                "symmetric_global_alloc<{state_tuple_type}>()".format(
                    state_tuple_type=self.state_tuple.getTupleTypename())

            init_template = self._cgenv.get_template('withoutkey_init.cpp')
            initializer = no_key_state_initializer
            func_name = self.func_name

        self.initializer = init_template.render(locals())
        state.addInitializers([self.initializer])

        if not hashtableInfo:
            self.input.produce(state)

        # now that everything is aggregated, produce the tuples

        # assert len(self.column_list()) == 1 \
        #    or isinstance(self.column_list()[0],
        #                  expression.AttributeRef), \
        #    """assumes first column is the key and second is aggregate result
#            column_list: %s""" % self.column_list()

        # add a dependence on the input aggregation pipeline
        for s in self.input_syncname:
            state.addToPipelinePropertySet('dependences', s)

        self._impl_produce(state)

    class AggregateSetter:

        def __init__(self, name, expression):
            self.name = name
            self.expression = expression

    def compile_assignments(self, assgns, inputTuple):
        """
        compile update statements
        """
        state_var_update_template = "auto {assignment};"
        state_var_updates = []
        state_vars = []
        decls = []
        inits = []

        for a in assgns:
            # doesn't have to use inputTuple.name,
            # but it will for simplicity
            # If it is a UDA, we need to compile the expression
            rhs = self.language().compile_expression(
                a.expression,
                # For resolving attributes of the consumed input tuple
                tupleref=inputTuple,
                # for resolving attributes of the state tuple
                state_scheme=self.aggregates_schema)

            # combine lhs, rhs with assignment
            code = "{lhs} = {rhs}".format(lhs=a.name, rhs=rhs[0])

            decls += rhs[1]
            inits += rhs[2]

            state_var_updates.append(
                state_var_update_template.format(assignment=code))
            state_vars.append(a.name)

        return state_var_updates, state_vars, decls, inits

    def _update_func(self, state, inputTuple):
        # add Builtins updaters and to those
        # from UDAs in self.updaters
        updaters_map = dict((k, (k, v)) for k, v in self.updaters)

        inp_sch = self.input.scheme()

        def aggregate_to_updater(index, aggr):
            if isinstance(aggr, aggregate.UdaAggregateExpression):
                attr_ref = aggr.input
                assert isinstance(
                    attr_ref, aggregate.NamedStateAttributeRef), \
                    "type(attr_ref)={0} but must be {1}".format(
                    type(attr_ref), aggregate.NamedStateAttributeRef.__class__)
                name, expr = updaters_map[attr_ref.name]
                return self.AggregateSetter(name, expr)
            elif isinstance(aggr, aggregate.BuiltinAggregateExpression):
                # get update function
                name = "_v{0}".format(index)
                input_type = self.language().typename(
                    aggr.input.typeof(inp_sch, None))
                op = aggr
                up_op_name = op.__class__.__name__
                state_type = self.language().typename(
                    aggr.typeof(
                        inp_sch,
                        None))
                update_func = \
                    "Aggregates::{op}<{state_type}, {input_type}>".format(
                        op=up_op_name,
                        state_type=state_type,
                        input_type=input_type)

                # create a class for this binary function so it can be compiled
                update_as_expression = expression.CustomBinaryFunction(
                    update_func,
                    # get the output type
                    self.aggregates_schema.get_types()[index],
                    # get the aggregate result attribute (left)
                    expression.NamedStateAttributeRef(
                        self.aggregates_schema.get_names()[index]),
                    # get the aggregate input attribute (right)
                    aggr.input)

                return self.AggregateSetter(name, update_as_expression)
            else:
                assert False, "expected every element of aggregate_list to " \
                              "be an aggregate"

        all_updaters = [
            aggregate_to_updater(
                i, a) for i, a in enumerate(
                self.aggregate_list)]

        update_updates, update_state_vars, update_decls, update_inits = \
            self.compile_assignments(all_updaters, inputTuple)

        update_def = self._cgenv.get_template(
            'update_definition.cpp').render(
            state_type=self.state_tuple.getTupleTypename(),
            input_type=inputTuple.getTupleTypename(),
            input_tuple_name=inputTuple.name,
            update_updates=update_updates,
            update_state_vars=update_state_vars,
            name=self.func_name)

        state.addDeclarations([update_def] + update_decls)
        state.addInitializers(update_inits)
        return update_state_vars

    def _init_def(self, state, inputTuple):
        # add Builtins inits and to those
        # from UDAs in self.inits
        inits_map = dict((k, (k, v)) for k, v in self.inits)
        inp_sch = self.input.scheme()

        def aggregate_to_init(index, aggr):
            if isinstance(aggr, aggregate.UdaAggregateExpression):
                attr_ref = aggr.input
                assert isinstance(
                    attr_ref, aggregate.NamedStateAttributeRef), \
                    "type(attr_ref)={0} but must be {1}".format(
                    type(attr_ref), aggregate.NamedStateAttributeRef.__class__)
                name, expr = inits_map[attr_ref.name]
                return self.AggregateSetter(name, expr)
            elif isinstance(aggr, aggregate.BuiltinAggregateExpression):
                name = "_v{0}".format(index)
                state_type = self.language().typename(
                    aggr.typeof(inp_sch, None))
                init_func = self._init_func_for_op(aggr) \
                    .format(state_type=state_type)
                return self.AggregateSetter(
                    name,
                    expression.CustomZeroaryFunction(
                        init_func,
                        # get the output type
                        self.aggregates_schema.get_types()[index]))

        all_initers = [
            aggregate_to_init(
                i, a) for i, a in enumerate(
                self.aggregate_list)]

        init_updates, init_state_vars, init_decls, init_inits = \
            self.compile_assignments(all_initers, inputTuple)

        init_def = self._cgenv.get_template('init_definition.cpp').render(
            state_type=self.state_tuple.getTupleTypename(),
            init_updates=init_updates,
            init_state_vars=init_state_vars,
            name=self.func_name)

        state.addDeclarations([init_def] + init_decls)
        state.addInitializers(init_inits)
        return init_state_vars

    def _combine_def(self, state, inputTuple):
        # add Builtins updaters and to those
        # from UDAs in self.updaters
        updaters_map = dict((k, (k, v)) for k, v in self.updaters)
        inp_sch = self.input.scheme()

        def aggregate_to_combiner(index, aggr):
            if isinstance(aggr, aggregate.UdaAggregateExpression):
                # TODO: Support decomposable aggregate state, instead of
                # TODO: just using the same function from self.updaters
                attr_ref = aggr.input
                assert isinstance(
                    attr_ref, aggregate.NamedStateAttributeRef), \
                    "type(attr_ref)={0} but must be {1}".format(
                    type(attr_ref), aggregate.NamedStateAttributeRef.__class__)
                name, expr = updaters_map[attr_ref.name]
                return self.AggregateSetter(name, expr)
            elif isinstance(aggr, aggregate.BuiltinAggregateExpression):
                # get combiner function
                name = "_v{0}".format(index)

                state_type = self.language().typename(
                    aggr.typeof(
                        inp_sch,
                        None))

                # hack to get the combiner function based on the aggregate
                # TODO: support decomposable state
                co_op_name = self._combiner_for_builtin_update(
                    aggr).__class__.__name__

                # type for left and right is state_type for combiner
                this_combine_func = "Aggregates::{op}<{type}, {type}>".format(
                    op=co_op_name, type=state_type)

                attribute0 = expression.NamedStateAttributeRef(
                    self.aggregates_schema.get_names()[index])
                attribute0.tagged_state_id = 0
                attribute1 = expression.NamedStateAttributeRef(
                    self.aggregates_schema.get_names()[index])
                attribute1.tagged_state_id = 1

                # create a class for this binary function so it can be compiled
                combine_as_expression = expression.CustomBinaryFunction(
                    this_combine_func,
                    # get the output type
                    self.aggregates_schema.get_types()[index],
                    # get the aggregate result attribute (left)
                    attribute0,
                    # get the aggregate input attribute (right)
                    attribute1)

                return self.AggregateSetter(name, combine_as_expression)
            else:
                assert False, "expected every element of aggregate_list to " \
                              "be an aggregate"

        all_combiners = [
            aggregate_to_combiner(
                i, a) for i, a in enumerate(
                self.aggregate_list)]

        combine_updates, combine_state_vars, combine_decls, combine_inits = \
            self.compile_assignments(all_combiners, inputTuple)

        # currently only needed for 0key reduce, since there is no key-based
        # reduce in the grappalang streaming aggregate
        combine_def = self._cgenv.get_template(
            'combine_definition.cpp').render(
            state_type=self.state_tuple.getTupleTypename(),
            combine_updates=combine_updates,
            combine_state_vars=combine_state_vars,
            name=self.func_name)

        state.addDeclarations([combine_def] + combine_decls)
        state.addInitializers(combine_inits)
        return combine_state_vars

    def _key_access_code(self, inputTuple, inp_sch):
        return [inputTuple.get_code(g.get_position(inp_sch))
                for g in self.grouping_list]

    def consume(self, inputTuple, fromOp, state):
        # save the inter-pipeline task name
        # Needs to be a list because could be multiple input occurences
        if not hasattr(self, 'input_syncname'):
            self.input_syncname = []
        self.input_syncname.append(get_pipeline_task_name(state))

        # not used, but must be resolved
        state.resolveSymbol(self.input_type_ref, inputTuple.getTupleTypename())

        inp_sch = self.input.scheme()

        update_vars = self._update_func(state, inputTuple)
        init_vars = self._init_def(state, inputTuple)

        assert set(update_vars) == set(init_vars), \
            """Initialized and update state vars are not the same \
            {0} != {1}""".format(update_vars, init_vars)

        if not self.useKey:
            # if 0key then add the combiner codes, otherwise
            # omit it because it may be invalid code
            combine_vars = self._combine_def(state, inputTuple)

            assert set(update_vars) == set(combine_vars), \
                """Combiner and update state vars are not the same \
                {0} != {1}""".format(update_vars,
                                     combine_vars)

        # generate the update and init function definitions
        state_tuple_decl = self.state_tuple.generateDefinition()

        # values for the materialize template (calling the update function)
        init_func = self.init_func
        update_func = self.update_func
        update_val = inputTuple.name
        input_type = inputTuple.getTupleTypename()

        if self.useKey:
            numkeys = len(self.grouping_list)
            keygets = self._key_access_code(inputTuple, inp_sch)

            materialize_template = self._cgenv.get_template('nkey_update.cpp')
        else:
            materialize_template = self._cgenv.get_template(
                'multi_uda_0key_update.cpp')

        if self.useKey:
            state.recordCodeWhenInLoop(self.language().comment("recycle") +
                                       "{hashname}->clear();\n".format(
                                           hashname=self._hashname))
        else:
            state.recordCodeWhenInLoop(self.language().comment("recycle") +
                                       self.initializer)

        hashname = self._hashname
        tuple_name = inputTuple.name
        pipeline_sync = state.getPipelineProperty("global_syncname")

        comment = self.language().comment("insert of " + str(self))

        code = materialize_template.render(locals())
        return code


def wait_statement(name):
    return GrappaLanguage.cgenv().get_template(
        'wait_statement.cpp').render(name=name)


def get_pipeline_task_name(state):
    name = "pipeline_{n}".format(n=state.getCurrentPipelineId())
    state.setPipelineProperty('sync', name)
    wait_stmt = wait_statement(name)
    state.addSeqWaitStatement(wait_stmt)
    return name


class GrappaHashJoin(GrappaJoin, GrappaOperator):
    _i = 0

    @staticmethod
    def __genHashName__():
        name = "hash_%03d" % GrappaHashJoin._i
        GrappaHashJoin._i += 1
        return name

    def __init__(self, *args):
        super(GrappaHashJoin, self).__init__(*args)
        self._cgenv = cppcommon.prepend_template_relpath(
            self.language().cgenv(),
            '{0}/hashjoin'.format(GrappaLanguage._template_path))

    def produce(self, state):
        declr_template = self._cgenv.get_template('hash_declaration.cpp')

        self.right.childtag = "right"

        my_sch = self.scheme()
        left_sch = self.left.scheme()
        right_sch = self.right.scheme()

        self.leftcols, self.rightcols = \
            algebra.convertcondition(self.condition,
                                     len(left_sch),
                                     left_sch + right_sch)

        keytype = CKeyUtils._aggregate_type(
            self.language(),
            right_sch,
            self.rightcols)

        # common index is defined by same right side and same key
        hashtableInfo = state.lookupExpr((self.right,
                                          frozenset(self.rightcols)))
        if not hashtableInfo:
            # if right child never bound then store hashtable symbol and
            # call right child produce
            self._hashname = GrappaHashJoin.__genHashName__()
            _LOG.debug("generate hashname %s for %s", self._hashname, self)

            hashname = self._hashname

            # declaration of hash map
            self.rightTupleTypeRef = state.createUnresolvedSymbol()
            in_tuple_type = self.rightTupleTypeRef.getPlaceholder()
            hashdeclr = declr_template.render(locals())
            state.addDeclarationsUnresolved([hashdeclr])

            init_template = self._cgenv.get_template('hash_init.cpp')

            self.initializer = init_template.render(locals())
            state.addInitializers([self.initializer])
            self.right.produce(state)
            state.saveExpr((self.right, frozenset(self.rightcols)),
                           (self._hashname, self.rightTupleTypename,
                            self.right_syncname))
            # TODO always safe here? I really want to call
            # TODO saveExpr before self.right.produce(),
            # TODO but I need to get the self.rightTupleTypename cleanly
        else:
            self.rightTupleTypeRef = None  # indicates CSE succeeded

            # if found a common subexpression on right child then
            # use the same hashtable
            self._hashname, self.rightTupleTypename, self.right_syncname\
                = hashtableInfo
            _LOG.debug("reuse hash %s for %s", self._hashname, self)

        self.left.childtag = "left"
        self.left.produce(state)

    def consume(self, t, src, state):
        if src.childtag == "right":
            comment = self.language().comment("right side of " + str(self))
            right_template = self._cgenv.get_template('insert_materialize.cpp')

            hashname = self._hashname
            keyname = t.name
            keyval = CKeyUtils._aggregate_val(t, self.rightcols)

            # Needs to be a list because could be multiple right sides
            # occurences
            if not hasattr(self, 'right_syncname'):
                self.right_syncname = []
            self.right_syncname.append(get_pipeline_task_name(state))

            self.rightTupleTypename = t.getTupleTypename()
            if self.rightTupleTypeRef is not None:
                state.resolveSymbol(self.rightTupleTypeRef,
                                    self.rightTupleTypename)

            pipeline_sync = state.getPipelineProperty('global_syncname')

            # materialization point
            code = right_template.render(locals())

            # recycling when right index is reused
            state.recordCodeWhenInLoop(
                self.language().comment("recycle") +
                "{hashname}.clear();\n".format(
                    hashname=hashname))

            return code

        if src.childtag == "left":
            comment = self.language().comment("left side of " + str(self))
            left_template = self._cgenv.get_template('lookup.cpp')

            # add a dependences on the right pipelines
            for s in self.right_syncname:
                state.addToPipelinePropertySet('dependences', s)

            hashname = self._hashname
            keyname = t.name
            input_tuple_type = t.getTupleTypename()
            keyval = CKeyUtils._aggregate_val(t, self.leftcols)

            pipeline_sync = state.getPipelineProperty('global_syncname')

            right_tuple_name = gensym()
            right_tuple_type = self.rightTupleTypename

            outTuple = GrappaStagedTupleRef(gensym(), self.scheme())
            out_tuple_type_def = outTuple.generateDefinition()
            out_tuple_type = outTuple.getTupleTypename()
            out_tuple_name = outTuple.name

            type1 = input_tuple_type
            type1numfields = len(t.scheme)
            type2 = right_tuple_type
            type2numfields = len(self.right.scheme())
            append_func_name, combine_function_def = \
                GrappaStagedTupleRef.get_append(
                    out_tuple_type,
                    type1, type1numfields,
                    type2, type2numfields)

            state.addDeclarations([out_tuple_type_def, combine_function_def])

            inner_plan_compiled = self.parent().consume(outTuple, self, state)

            code = left_template.render(locals())
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


class GrappaSelect(cppcommon.CBaseSelect, GrappaOperator):
    pass


# Basic apply like serial C++
class GrappaApply(cppcommon.CBaseApply, GrappaOperator):
    pass


# Basic duplication based bag union like serial C++
class GrappaUnionAll(cppcommon.CBaseUnionAll, GrappaOperator):
    pass


# Basic materialized copy based project like serial C++
class GrappaProject(cppcommon.CBaseProject, GrappaOperator):
    pass


class GrappaSink(cppcommon.CBaseSink, GrappaOperator):
    pass


class GrappaFileScan(cppcommon.CBaseFileScan, GrappaOperator):

    def new_tuple_ref_for_filescan(self, resultsym, scheme):
        # make new tuple 64-byte aligned if scanning into a global array
        aligned = self.array_representation == \
            _ARRAY_REPRESENTATION.GLOBAL_ARRAY
        return GrappaStagedTupleRef(resultsym, scheme,
                                    aligned=aligned)

    def __init__(self, representation=_ARRAY_REPRESENTATION.GLOBAL_ARRAY,
                 relation_key=None, _scheme=None, cardinality=None):
        self.array_representation = representation
        super(GrappaFileScan, self).__init__(
            relation_key, _scheme, cardinality)

    def __get_ascii_scan_template__(self):
        _LOG.warn("binary/ascii is command line choice")
        template_name = {
            _ARRAY_REPRESENTATION.GLOBAL_ARRAY: 'file_scan.cpp',
            _ARRAY_REPRESENTATION.SYMMETRIC_ARRAY:
                'symmetric_array_file_scan.cpp'
        }[self.array_representation]
        return self._language.cgenv().get_template(template_name)

    def __get_binary_scan_template__(self):
        _LOG.warn("binary/ascii is command line choice")
        template_name = {
            _ARRAY_REPRESENTATION.GLOBAL_ARRAY: 'file_scan.cpp',
            _ARRAY_REPRESENTATION.SYMMETRIC_ARRAY:
                'symmetric_array_file_scan.cpp'
        }[self.array_representation]
        return self._language.cgenv().get_template(template_name)

    def __get_relation_decl_template__(self, name):
        template_name = {
            _ARRAY_REPRESENTATION.GLOBAL_ARRAY:
                'global_array_relation_declaration.cpp',
            _ARRAY_REPRESENTATION.SYMMETRIC_ARRAY:
                'symmetric_array_relation_declaration.cpp'
        }[self.array_representation]

        return self._language.cgenv().get_template(template_name)

    def _get_input_aux_decls_template(self):
        return self._language.cgenv().get_template(
            'input_relation_declarations.cpp')

    def __repr__(self):
        return "{op}({rep!r}, {rk!r}, {sch!r}, {card!r})".format(
            rep=self.array_representation,
            op=self.opname(), rk=self.relation_key, sch=self._scheme,
            card=self._cardinality)


class GrappaStore(cppcommon.CBaseStore, GrappaOperator):

    def __file_code__(self, t, state):
        my_sch = self.scheme()

        filename = (str(self.relation_key).split(":")[2])
        outputnamedecl = define_cl_arg(
            'string',
            'output_file',
            '"{0}"'.format(filename),
            "Output File")

        state.addDeclarations([outputnamedecl])
        names = [x.encode('UTF8') for x in my_sch.get_names()]
        schemefile = \
            'writeSchema("{s}", FLAGS_output_file+".scheme");\n'.format(s=zip(
                names, my_sch.get_types()))
        state.addPreCode(schemefile)
        resultfile = 'writeTuplesUnordered(&result, FLAGS_output_file+".bin");'
        state.addPipelineFlushCode(resultfile)

        return ""


class TempRelationSymbol:

    def __init__(self, name):
        self._name = name

    def symbol(self):
        return "_temp_table_{}".format(self._name)

    def __eq__(self, other):
        return isinstance(other, TempRelationSymbol) \
            and self._name == other._name


class GrappaStoreTemp(algebra.StoreTemp, GrappaOperator):

    def produce(self, state):
        temp_rel = TempRelationSymbol(self.name)

        # Recycle cannot happen for every visit (in consume) because
        # it will get called with binary streaming operators.
        # Calling it here is safe because all pipelines below
        # the StoreTemp should not be using this temp relation
        #
        # An alternative approach to consider in the future is,
        # to just put recycle in consume() but require a barrier
        # after it that involves all pipelines this StoreTemp appears in

        self.num_producers = state.createUnresolvedSymbol()
        state.addCode(
            self._language.cgenv().get_template(
                'symmetric_array_temprelation_recycle.cpp').render(
                sym=temp_rel.symbol(),
                num_producers=self.num_producers.getPlaceholder()))

        self.input.produce(state)

    def consume(self, t, src, state):
        temp_rel = TempRelationSymbol(self.name)

        # uses symmetric array representation only

        tuple = state.lookupTupleDef(temp_rel.symbol())
        if not tuple:
            tuple = self.new_tuple_ref(temp_rel.symbol(), self.scheme())
            state.saveTupleDef(temp_rel.symbol(), tuple)

            declTupleType = tuple.generateDefinition()
            declRelation = self._language.cgenv()\
                .get_template('symmetric_array_temprelation_declaration.cpp')\
                .render(tuple_type=tuple.getTupleTypename(),
                        resultsym=temp_rel.symbol())

            state.addDeclarations([declTupleType,
                                   declRelation])

            init_stmt = self._language.cgenv()\
                .get_template('symmetric_array_temprelation_init.cpp')\
                .render(sym=temp_rel.symbol(),
                        tuple_type=tuple.getTupleTypename())

            state.addInitializers([init_stmt])

        newtuple = tuple.copy_type()
        state.addDeclarations([newtuple.generateDefinition()])
        assignment_code = \
            cppcommon.createTupleTypeConversion(self.language(),
                                                state,
                                                t,
                                                newtuple)

        state.addPostCode(self._language.cgenv()
                          .get_template(
            'symmetric_array_temprelation_materializer_done.cpp')
            .render(sym=temp_rel.symbol()))

        state.resolveCounterSymbol(self.num_producers)

        return assignment_code + self.language().comment(self) + self._language.cgenv()\
            .get_template('symmetric_array_temprelation_materialize.cpp')\
            .render(sym=temp_rel.symbol(), input_tuple_name=newtuple.name)


class GrappaNullInput(cppcommon.CBaseFileScan, GrappaOperator):

    def __get_binary_scan_template__(self):
        return None

    def __get_ascii_scan_template__(self):
        return None

    def produce(self, state):
        sym = TempRelationSymbol(self.relation_key)
        assert state.lookupTupleDef(sym.symbol()) is not None, \
            "Expected {} to already exist".format(sym.symbol())
        self.parent().consume(sym, self, state)

    def consume(self, t, src, state):
        assert False, "as a source, no need for consume"


class GrappaShuffle(algebra.Shuffle, GrappaOperator):

    def produce(self, state):
        self.input.produce(state)

    def consume(self, t, src, state):
        inner_plan_compiled = self.parent().consume(t, self, state)

        columnlist_nums = [c.position for c in self.columnlist]

        code = self.language().cgenv().get_template('shuffle.cpp').render(
            keyname=t.name,
            keytype=CKeyUtils._aggregate_type(
                self.language(),
                self.input.scheme(),
                columnlist_nums),
            keyval=CKeyUtils._aggregate_val(
                t,
                columnlist_nums),
            pipeline_sync=state.getPipelineProperty('global_syncname'),
            comment=self.language().comment(
                self.shortStr()),
            inner_code=inner_plan_compiled)

        return code


class MemoryScanOfFileScan(rules.Rule):

    def __init__(self, array_rep, memory_scan_class=GrappaMemoryScan):
        self._array_rep = array_rep
        self._memory_scan_class = memory_scan_class
        super(MemoryScanOfFileScan, self).__init__()

    """A rewrite rule for making a scan into materialization
     in memory then memory scan"""

    def fire(self, expr):
        if isinstance(expr, algebra.Scan) \
                and not isinstance(expr, GrappaFileScan):
            return self._memory_scan_class(
                GrappaFileScan(self._array_rep,
                               expr.relation_key,
                               expr.scheme()),
                self._array_rep)
        return expr

    def __str__(self):
        return "Scan => MemoryScan(FileScan) [{0}]".format(self._array_rep)


class ScanTempToMemoryScan(rules.Rule):

    """Logical ScanTemp to MemoryScan"""

    def fire(self, expr):
        if isinstance(expr, algebra.ScanTemp):
            return GrappaMemoryScan(GrappaNullInput(expr.name, expr.scheme()),
                                    _ARRAY_REPRESENTATION.SYMMETRIC_ARRAY,
                                    expr.name,
                                    expr.analyzed_num_tuples
                                    if hasattr(expr, 'analyzed_num_tuples')
                                    else 10000)

        return expr


class GrappaPartitionGroupBy(GrappaGroupBy):

    def __init__(self, *args):
        super(GrappaPartitionGroupBy, self).__init__(*args)
        # Override some GrappaGroupBy template with new ones.
        # The rest of the generation logic is the same
        self._cgenv = cppcommon.prepend_template_relpath(
            self._cgenv, '{0}/partition_groupby'.format(
                GrappaLanguage._template_path))


class GrappaBroadcastCrossProduct(algebra.CrossProduct, GrappaOperator):

    def produce(self, state):

        self.right.childtag = "right"
        self.right.produce(state)

        self.left.childtag = "left"
        self.left.produce(state)

    def _declare_broadcast_tuple(self, t, state):
        # declare global var and broadcast value
        self.broadcast_tuple = t.copy_type()
        var_decl = self.language().cgenv().get_template(
            'tuple_declaration.cpp').render(
            dst_type_name=self.broadcast_tuple.getTupleTypename(),
            dst_name=self.broadcast_tuple.name)
        state.addDeclarations([var_decl])

    def consume(self, t, src, state):
        if src.childtag == "right":
            # right to left dependency
            self.right_syncname = get_pipeline_task_name(state)

            code = self.language().comment(self.shortStr() + " RIGHT")

            self._declare_broadcast_tuple(t, state)

            code += """on_all_cores([=] {{
                  {global_name} = {input_name};
                   }});
                   """.format(global_name=self.broadcast_tuple.name,
                              input_name=t.name)

            return code

        elif src.childtag == "left":
            # right to left dependency
            state.addToPipelinePropertySet('dependences', self.right_syncname)

            code = self.language().comment(self.shortStr() + " LEFT")

            # add global field to your tuple

            output = GrappaStagedTupleRef(gensym(), self.scheme())

            type1 = t.getTupleTypename()
            type1numfields = len(t.scheme)
            type2 = self.broadcast_tuple.getTupleTypename()
            type2numfields = len(self.broadcast_tuple.scheme)
            append_func_name, combine_function_def = \
                GrappaStagedTupleRef.get_append(
                    output.getTupleTypename(),
                    type1, type1numfields,
                    type2, type2numfields)

            state.addDeclarations([output.generateDefinition(),
                                   combine_function_def])

            code += """
            {out_tuple_type} {out_tuple_name} =
              {append_func_name}({left_name}, {right_name});
              """.format(out_tuple_type=output.getTupleTypename(),
                         out_tuple_name=output.name,
                         append_func_name=append_func_name,
                         left_name=t.name,
                         right_name=self.broadcast_tuple.name)

            inner_plan_compiled = self.parent().consume(output, self, state)

            return code + inner_plan_compiled

        else:
            assert False, "bad childtag: {0}".format(src.childtag)


class Iterator(object):

    def operator_code(self, **kwargs):
        return self.iter_cgenv().get_template(
            "instantiate_operator.cpp").render(kwargs)

    def default_operator_code(self, **kwargs):
        kwargs['call_constructor'] = "{class_symbol}({inputsym})".format(
            **kwargs)
        return self.operator_code(**kwargs)

    def sink_operator_code(self, **kwargs):
        return self.iter_cgenv().get_template(
            "instantiate_sink.cpp").render(kwargs)

    def declare_sink(self, state, cgenv=None):
        if not cgenv:
            cgenv = self.iter_cgenv()
        symbol = "frag_{0}".format(gensym())
        state.setPipelineProperty('fragment_symbol', symbol)
        state.addDeclarations(
            [cgenv.get_template("sink_declaration.cpp").render(symbol=symbol)])
        return symbol

    def assign_symbol(self):
        self.symbol = gensym()

    def iter_cgenv(cls, env=GrappaLanguage.cgenv()):
        return cppcommon.prepend_template_relpath(
            env, '{0}/iterators/'.format(GrappaLanguage._template_path))


class IGrappaMemoryScan(GrappaMemoryScan, Iterator):

    def _constructor(self, inputsym, state):
        return "Scan<{type}>({inputsym})".format(
            type=state.lookupTupleDef(inputsym).getTupleTypename(),
            inputsym=inputsym)

    def consume(self, inputsym, src, state):
        _ = create_pipeline_synchronization(state)
        _ = get_pipeline_task_name(state)

        self.assign_symbol()
        stagedTuple = state.lookupTupleDef(inputsym)
        state.addOperator(self.operator_code(
            produce_type=stagedTuple.getTupleTypename(),
            symbol=self.symbol,
            call_constructor=self._constructor(inputsym, state)
        ))
        self.parent().consume(stagedTuple, self, state)

        state.addPipeline()
        return None


class IGrappaSelect(GrappaSelect, Iterator):

    def consume(self, t, src, state):
        class_symbol = "Select_{sym}".format(sym=gensym())
        self.assign_symbol()
        state.addDeclarations([self.iter_cgenv().get_template(
            "select.cpp").render(
            class_symbol=class_symbol,
            consume_type=t.getTupleTypename(),
            produce_type=t.getTupleTypename(),
            consume_tuple_name=t.name,
            expression=self._compile_condition(t, state)
        )])
        state.addOperator(self.default_operator_code(
            produce_type=t.getTupleTypename(),
            symbol=self.symbol,
            inputsym=src.symbol,
            class_symbol=class_symbol
        ))
        self.parent().consume(t, self, state)
        return None


class IGrappaStore(GrappaStore, Iterator):

    def _constructor(self, t, inputsym, state):
        return "Store<{type}>({inputsym}, &result)".format(
            type=t.getTupleTypename(),
            inputsym=inputsym)

    def consume(self, t, src, state):
        symbol = self.declare_sink(state)
        self._add_result_declaration(t, state)
        state.addOperator(self.sink_operator_code(
            symbol=symbol,
            call_constructor=self._constructor(t, src.symbol, state)
        ))
        return None


class IGrappaApply(GrappaApply, Iterator):

    def consume(self, t, src, state):
        class_symbol = "Apply_{}".format(gensym())
        self.symbol = gensym()

        state.addDeclarations([self.iter_cgenv().get_template(
            "apply.cpp").render(
            class_symbol=class_symbol,
            produce_type=self.newtuple.getTupleTypename(),
            consume_type=t.getTupleTypename(),
            produce_tuple_name=self.newtuple.name,
            consume_tuple_name=t.name,
            statements=self._apply_statements(t, state)
        )])
        state.addOperator(self.default_operator_code(
            produce_type=self.newtuple.getTupleTypename(),
            class_symbol=class_symbol,
            symbol=self.symbol,
            inputsym=src.symbol
        ))

        self.parent().consume(self.newtuple, self, state)

        return None


class IGrappaHashJoin(GrappaSymmetricHashJoin, Iterator):

    def _constructor(self, inputsym, class_symbol):
        return "{class_symbol}(&{hashname}, {inputsym})".format(
            class_symbol=class_symbol,
            hashname=self._hashname,
            inputsym=inputsym
        )

    def produce(self, state):
        super(IGrappaHashJoin, self).produce(state)

        _ = create_pipeline_synchronization(state)
        _ = get_pipeline_task_name(state)

        # add dependences on left and right inputs
        state.addToPipelinePropertySet('dependences', self.right_syncname)
        state.addToPipelinePropertySet('dependences', self.left_syncname)

        self.assign_symbol()
        class_symbol = "HashJoinSource_{sym}".format(sym=gensym())
        keytype = CKeyUtils._aggregate_type(
            self.language(),
            self.right.scheme(),
            self.rightcols)
        left_type = self.leftTypeRef.getPlaceholder()
        right_type = self.rightTypeRef.getPlaceholder()
        out_tuple_type = self.outTuple.getTupleTypename()

        append_func_name, combine_function_def = \
            GrappaStagedTupleRef.get_append(
                out_tuple_type,
                left_type, len(self.left.scheme()),
                right_type, len(self.right.scheme()))

        state.addDeclarations([combine_function_def])

        state.addDeclarations([self.iter_cgenv().get_template(
            "hashjoin_source.cpp").render(
            class_symbol=class_symbol,
            keytype=keytype,
            left_tuple_type=left_type,
            right_tuple_type=right_type,
            out_tuple_type=out_tuple_type,
            left_name=gensym(),
            right_name=gensym(),
            append_func_name=append_func_name
        )])

        state.addOperator(self.operator_code(
            produce_type=out_tuple_type,
            symbol=self.symbol,
            call_constructor="{class_symbol}(&{hashname})".format(
                class_symbol=class_symbol,
                hashname=self._hashname)
        ))

        self.parent().consume(self.outTuple, self, state)
        state.addPipeline()

    def consume(self, t, src, state):
        class_symbol_template = "HashJoinSink{side}_{sym}"
        symbol = self.declare_sink(state)

        if src.childtag == 'right':
            side = 'Right'
            class_symbol = class_symbol_template.format(
                side=side,
                sym=gensym())

            keyval = CKeyUtils._aggregate_val(t, self.rightcols)
            keytype = CKeyUtils._aggregate_type(
                self.language(),
                self.right.scheme(),
                self.rightcols)
            state.resolveSymbol(self.rightTypeRef, t.getTupleTypename())

            # save to add after left type added
            self.right_class_decl = self.iter_cgenv().get_template(
                "hashjoin_sink.cpp").render(
                class_symbol=class_symbol,
                side=side,
                keytype=keytype,
                left_tuple_type=self.leftTypeRef.getPlaceholder(),
                right_tuple_type=t.getTupleTypename(),
                input_tuple_name=t.name,
                keyval=keyval,
                pipeline_sync=state.getPipelineProperty("global_syncname"))

            self.right_syncname = get_pipeline_task_name(state)

        elif src.childtag == 'left':
            side = 'Left'
            class_symbol = class_symbol_template.format(
                side=side,
                sym=gensym())

            keyval = CKeyUtils._aggregate_val(t, self.leftcols)
            keytype = CKeyUtils._aggregate_type(
                self.language(),
                self.left.scheme(),
                self.leftcols)
            state.resolveSymbol(self.leftTypeRef, t.getTupleTypename())

            state.addDeclarations([self.iter_cgenv().get_template(
                "hashjoin_sink.cpp").render(
                class_symbol=class_symbol,
                side=side,
                keytype=keytype,
                left_tuple_type=t.getTupleTypename(),
                right_tuple_type=self.rightTypeRef.getPlaceholder(),
                input_tuple_name=t.name,
                keyval=keyval,
                pipeline_sync=state.getPipelineProperty("global_syncname")
                # add right side here now that left type declared
            ), self.right_class_decl])

            self.left_syncname = get_pipeline_task_name(state)

        else:
            assert False, "Invalid child tag: {}".format(src.childtag)

        state.addOperator(self.sink_operator_code(
            symbol=symbol,
            call_constructor=self._constructor(src.symbol, class_symbol)
        ))

        return None


class IGrappaGroupBy(GrappaGroupBy, Iterator):

    def __init__(self, *args):
        super(IGrappaGroupBy, self).__init__(*args)
        self._igroupby_cgenv = self.iter_cgenv(env=self._cgenv)

    def _reuse_properties(self, other):
        super(IGrappaGroupBy, self)._reuse_properties(other)
        # save additional properties
        self.input_tuple_type = other.input_tuple_type

    def _impl_produce(self, state):
        pipeline_sync = create_pipeline_synchronization(state)
        get_pipeline_task_name(state)

        if self.useKey:
            produce_template = self._igroupby_cgenv.get_template(
                "multikey_groupby_source.cpp")
            class_symbol = "AggregateSource_{}".format(gensym())
        else:
            produce_template = self._igroupby_cgenv.get_template(
                "0key_groupby_source.cpp")
            class_symbol = "ZeroKeyAggregateSource_{}".format(gensym())

        output_tuple = GrappaStagedTupleRef(gensym(), self.scheme())
        state.addDeclarations([output_tuple.generateDefinition()])

        assignment_code = self._assignment_code(output_tuple)

        self.assign_symbol()

        inp_sch = self.input.scheme()

        state.addDeclarations([produce_template.render(
            class_symbol=class_symbol,
            produce_type=output_tuple.getTupleTypename(),
            produce_tuple_name=output_tuple.name,
            keytype=self._key_type(inp_sch),
            state_type=self.state_tuple.getTupleTypename(),
            input_type=self.input_tuple_type,
            assignment_code=assignment_code,
            mapping_var_name=output_tuple.name + "_entry",
            combine_func=self.combine_func,
            state_tuple_name=output_tuple.name + "_tmp"
        )])

        state.addOperator(
            self.operator_code(
                produce_type=output_tuple.getTupleTypename(),
                symbol=self.symbol,
                call_constructor="{class_symbol}({hashname})".format(
                    class_symbol=class_symbol,
                    hashname=self._hashname)))

        self.parent().consume(output_tuple, self, state)

        state.setPipelineProperty("type", "in_memory")
        state.addPipeline()

    def _init_template(self):
        return self._igroupby_cgenv.get_template('withkey_init.cpp')

    def consume(self, t, src, state):
        # save the inter-pipeline task name
        if not hasattr(self, 'input_syncname'):
            self.input_syncname = []
        self.input_syncname.append(get_pipeline_task_name(state))

        self.input_tuple_type = t.getTupleTypename()

        state.resolveSymbol(self.input_type_ref, self.input_tuple_type)

        inp_sch = self.input.scheme()

        self._update_func(state, t)
        self._init_def(state, t)

        # generate class definition, if needed
        if self.useKey:
            class_symbol = "AggregateSink_{}".format(gensym())
            state.addDeclarations([
                self._igroupby_cgenv.get_template(
                    "multikey_groupby_sink.cpp").render(
                    class_symbol=class_symbol,
                    consume_type=t.getTupleTypename(),
                    keytype=self._key_type(inp_sch),
                    state_type=self.state_tuple.getTupleTypename(),
                    pipeline_sync=state.getPipelineProperty("global_syncname"),
                    consume_tuple_name=t.name,
                    keygets=self._key_access_code(t, inp_sch)
                )
            ])
        else:
            self._combine_def(state, t)

        symbol = self.declare_sink(state, cgenv=self._igroupby_cgenv)

        # generate instantiation
        if self.useKey:
            state.addOperator(
                self.sink_operator_code(
                    symbol=symbol,
                    call_constructor="""
                    {class_symbol}({inputsym}, {hashname})""".format(
                        inputsym=src.symbol,
                        class_symbol=class_symbol,
                        hashname=self._hashname)))
        else:
            state.addOperator(
                self.sink_operator_code(
                    symbol=symbol,
                    call_constructor="""
                    ZeroKeyAggregateSink<{consume_type}, {state_type}>(
                    {inputsym}, {hashname}, &{update_func}, &{init_func})
                    """.format(
                        inputsym=src.symbol,
                        hashname=self._hashname,
                        update_func=self.update_func,
                        init_func=self.init_func,
                        consume_type=t.getTupleTypename(),
                        state_type=self.state_tuple.getTupleTypename())))

        return None


class IGrappaPartitionGroupBy(IGrappaGroupBy):

    def __init__(self, *args):
        super(IGrappaPartitionGroupBy, self).__init__(*args)
        # Override some IGrappaGroupBy template with new ones.
        # The rest of the generation logic is the same
        self._igroupby_cgenv = cppcommon.prepend_template_relpath(
            self._igroupby_cgenv,
            '{0}/iterators/partition_groupby'.format(
                GrappaLanguage._template_path))


class IGrappaShuffle(algebra.Shuffle, GrappaOperator, Iterator):

    def __init__(self, *args):
        raise NotImplementedError(
            "Should not be necessary currently, see {0}".format(
                FuseGroupByShuffle))

    def produce(self, state):
        self.input.produce(state)

    def consume(self, t, src, state):
        # currently a no-op because IGroupBy takes care of the shuffle
        self.parent().consume(t, src, state)
        return None


class IGrappaBroadcastCrossProduct(GrappaBroadcastCrossProduct, Iterator):

    def consume(self, t, src, state):
        if src.childtag == "right":
            symbol = self.declare_sink(state)
            # right to left dependency
            self.right_syncname = get_pipeline_task_name(state)
            self._declare_broadcast_tuple(t, state)
            state.addOperator(
                self.sink_operator_code(
                    symbol=symbol,
                    call_constructor="""BroadcastTupleSink<{consume_type}>
                    ({inputsym}, &{broadcast_tuple})""".format(
                        consume_type=t.getTupleTypename(),
                        inputsym=src.symbol,
                        broadcast_tuple=self.broadcast_tuple.name)))
            return None

        elif src.childtag == "left":
            # right to left dependency
            state.addToPipelinePropertySet('dependences', self.right_syncname)
            output = GrappaStagedTupleRef(gensym(), self.scheme())

            append_func_name, combine_function_def = \
                GrappaStagedTupleRef.get_append(
                    output.getTupleTypename(),
                    t.getTupleTypename(),
                    len(t.scheme),
                    self.broadcast_tuple.getTupleTypename(),
                    len(self.broadcast_tuple.scheme))

            class_symbol = "BroadcastTupleStream_{}".format(gensym())

            class_decl = self.iter_cgenv().get_template(
                'broadcast_stream.cpp').render(
                class_symbol=class_symbol,
                left_type=t.getTupleTypename(),
                right_type=self.broadcast_tuple.getTupleTypename(),
                output_type=output.getTupleTypename(),
                output_name=output.name,
                append_func_name=append_func_name)

            state.addDeclarations([output.generateDefinition(),
                                   combine_function_def,
                                   class_decl])

            self.assign_symbol()
            state.addOperator(
                self.operator_code(
                    produce_type=output.getTupleTypename(),
                    symbol=self.symbol,
                    call_constructor="""
                    {class_symbol}({inputsym}, &{broadcast_tuple})""".format(
                        class_symbol=class_symbol,
                        broadcast_tuple=self.broadcast_tuple.name,
                        inputsym=src.symbol)))

            self.parent().consume(output, self, state)
            return None

        else:
            assert False, "bad childtag: {0}".format(src.childtag)


class GrappaSequence(algebra.Sequence, GrappaOperator):

    def produce(self, state):
        for c in self.children():
            c.produce(state)

            # During c.produce, the compiler will be adding
            # wait statements for each pipeline in the child c.
            # To enforce sequential execution of the
            # sequence, we need to end each child with
            # waits for its pipelines
            waits = state.getAndFlushSeqWaitStatements()
            for s in waits:
                state.addCode(s)

    def consume(self, sym, src, state):
        raise NotImplementedError(
            "GrappaSequence is a statement not expression. Its children"
            "should not call consume")


class GrappaDoWhile(algebra.DoWhile, GrappaOperator):

    def produce(self, state):
        state.addDeclarations(["bool continue_while;"])

        state.addCode("""
        WhileLoopManager while_manager;
        do {
        while_manager.iteration_start();
        continue_while = false;
        auto continue_while_a = make_global(&continue_while);
        """)

        state.enterLoop()

        for c in self.args:  # condition (args[-1]) is not treated specially
            c.produce(state)

            # See GrappaSequence.produce() for explanation
            waits = state.getAndFlushSeqWaitStatements()
            for s in waits:
                state.addCode(s)

        # put the recycle code before the pipeline code
        # within the loop, pipelines are queued instead of emitted
        pipes, recycles = state.exitLoop()
        for c in recycles:
            state.addCode(c)
        for c in pipes:
            state.addCode(c)

        state.addCode("""
            while_manager.iteration_end();
        } while (continue_while);
        """)

    def consume(self, sym, src, state):
        raise NotImplementedError(
            "GrappaDoWhile is a statement not expression. Its children"
            "should not call consume")


class GrappaTest(algebra.UnaryOperator, GrappaOperator):

    """Sets the found "register" to true if relation is
    non-empty"""

    def produce(self, state):
        self.input.produce(state)

    def consume(self, t, src, state):
        return """delegate::write(continue_while_a, {get});
        """.format(get=t.get_code(0))

    def num_tuples(self):
        return self.input.num_tuples()

    def partitioning(self):
        return self.input.partitioning()

    def shortStr(self):
        return "{op}".format(op=self.opname())


class GrappaWhileCondition(rules.Rule):

    """In the DoWhile condition, insert a test operator to check
    result of the condition table"""

    def fire(self, expr):
        if isinstance(expr, algebra.DoWhile):
            newdw = rules.OneToOne(algebra.DoWhile, GrappaDoWhile).fire(expr)
            newdw.copy(expr)
            newdw.children()[-1] = GrappaTest(expr.children()[-1])
            return newdw
        return expr

    def __str__(self):
        return """DoWhile[..., Cond] => \
        GrappaDoWhile[..., GrappaTest[Cond]]"""


class DistinctToGroupby(rules.Rule):
    """Change distinct back to Groupby(attributes)"""

    def __init__(self, groupby_class):
        self.gb_class = groupby_class

    def fire(self, expr):
        if isinstance(expr, algebra.Distinct):
            return self.gb_class([expression.UnnamedAttributeRef(i)
                                  for i in range(len(expr.scheme()))],
                                 [],
                                 expr.input)
        else:
            return expr

    def __str__(self):
        return """Distinct => Groupby(all attributes, no aggregate)"""


class CrossProductWithSmall(rules.Rule):

    """
    If there is a cross product between a relation and
    a singleton, then broadcast the singleton: cross product
    is just adding an attribute to every tuple from the other relation
    """

    def __init__(
            self,
            broadcast_crossproduct_class=GrappaBroadcastCrossProduct):
        self.op_class = broadcast_crossproduct_class

    def fire(self, expr):
        if isinstance(expr, algebra.CrossProduct):
            if expr.right.num_tuples() == 1:
                return self.op_class(expr.left, expr.right)

        return expr

    def __str__(self):
        return "CrossProduct(big, singleton) => " \
               "{}(big, singleton)".format(self.op_class.__name__)


class FuseGroupByShuffle(rules.Rule):

    """
    Global groupby is a fusion of a shuffle and a groupby
    """

    def __init__(self, global_groupby_class):
        self.global_groupby_class = global_groupby_class

    def fire(self, expr):
        if isinstance(
                expr,
                algebra.GroupBy) and isinstance(
                expr.input,
                algebra.Shuffle):
            # make sure the shuffle is for the groupby
            if expr.grouping_list == expr.input.columnlist:
                x = rules.OneToOne(
                    algebra.GroupBy,
                    self.global_groupby_class).fire(expr)
                x.input = expr.input.input
                return x
            else:
                _LOG.warning(
                    "missed {0} because the shuffle isn't for the groupby")
        else:
            return expr

    def __str__(self):
        return "GroupBy($a..$z)[Shuffle($a..$z)] => {0}($a..$z)".format(
            self.global_groupby_class)


def iteratorfy(emit_print, scan_array_repr, groupby_class):
    return [
        FuseGroupByShuffle(IGrappaGroupBy),

        rules.ProjectingJoinToProjectOfJoin(),

        rules.OneToOne(algebra.Select, IGrappaSelect),
        MemoryScanOfFileScan(scan_array_repr, IGrappaMemoryScan),
        rules.OneToOne(algebra.Apply, IGrappaApply),
        rules.OneToOne(algebra.Join, IGrappaHashJoin),
        # rules.OneToOne(algebra.Shuffle, IGrappaShuffle),
        # rules.OneToOne(algebra.Project, GrappaProject),
        # rules.OneToOne(algebra.UnionAll, GrappaUnionAll),
        cppcommon.StoreToBaseCStore(emit_print, IGrappaStore),
        CrossProductWithSmall(IGrappaBroadcastCrossProduct),
        DistinctToGroupby(groupby_class)
    ]


def grappify(join_type, emit_print,
             scan_array_repr, groupby_class):
    if isinstance(join_type, str):
        join_type_class = dict(
            (c.__name__,
             c) for c in [
                GrappaHashJoin,
                GrappaSymmetricHashJoin,
                GrappaShuffleHashJoin,
                GrappaOverSynchronizedSymmetricHashJoin])[join_type]
    else:
        join_type_class = join_type

    return [
        rules.ProjectingJoinToProjectOfJoin(),

        rules.OneToOne(algebra.Select, GrappaSelect),
        MemoryScanOfFileScan(scan_array_repr),
        rules.OneToOne(algebra.Apply, GrappaApply),
        rules.OneToOne(algebra.Join, join_type_class),
        rules.OneToOne(algebra.Project, GrappaProject),
        rules.OneToOne(algebra.Shuffle, GrappaShuffle),
        rules.OneToOne(algebra.UnionAll, GrappaUnionAll),
        cppcommon.StoreToBaseCStore(emit_print, GrappaStore),
        CrossProductWithSmall(),
        rules.OneToOne(algebra.Sink, GrappaSink),
        rules.OneToOne(algebra.StoreTemp, GrappaStoreTemp),
        ScanTempToMemoryScan(),
        rules.OneToOne(algebra.Sequence, GrappaSequence),
        rules.OneToOne(algebra.DoWhile, GrappaDoWhile),
        rules.OneToOne(algebra.SingletonRelation, GrappaSingletonRelation),
        DistinctToGroupby(groupby_class)

        # Don't need this because we support two-key
        # cppcommon.BreakHashJoinConjunction(GrappaSelect, join_type)
    ]


class GrappaAlgebra(Algebra):

    def __init__(self, emit_print=cppcommon.EMIT_CONSOLE):
        self.emit_print = emit_print

    def opt_rules(self,
                  join_type=GrappaHashJoin,
                  scan_array_repr=_ARRAY_REPRESENTATION.GLOBAL_ARRAY,
                  compiler='push',
                  SwapJoinSides=False,
                  external_indexing=False,
                  **kwargs):

        # datalog_rules = [
        # rules.removeProject(),
        #     rules.CrossProduct2Join(),
        #     rules.SimpleGroupBy(),
        # SwapJoinSides(),
        #     rules.OneToOne(algebra.Select, GrappaSelect),
        #     rules.OneToOne(algebra.Apply, GrappaApply),
        # rules.OneToOne(algebra.Scan,MemoryScan),
        #     MemoryScanOfFileScan(),
        # rules.OneToOne(algebra.Join, GrappaSymmetricHashJoin),
        #     rules.OneToOne(algebra.Join, self.join_type),
        #     rules.OneToOne(algebra.Project, GrappaProject),
        #     rules.OneToOne(algebra.GroupBy, GrappaGroupBy),
        # TODO: this Union obviously breaks semantics
        #     rules.OneToOne(algebra.Union, GrappaUnionAll),
        #     rules.OneToOne(algebra.Store, GrappaStore)
        # rules.FreeMemory()
        # ]

        if compiler == 'push':
            groupby_classes = {
                'global': GrappaGroupBy,
                'partition': GrappaPartitionGroupBy}
        elif compiler == 'iterator':
            groupby_classes = {
                'global': IGrappaGroupBy,
                'partition': IGrappaPartitionGroupBy}
        else:
            raise ValueError(
                "unsupported argument compiler={0}".format(compiler))

        groupby_sematics = kwargs.get('groupby_semantics', 'global')
        if groupby_sematics == 'global':
            groupby_rules = [
                rules.OneToOne(
                    algebra.GroupBy,
                    groupby_classes['global'])]
            groupby_class = groupby_classes['global']
        elif groupby_sematics == 'partition':
            groupby_rules = rules.distributed_group_by(
                groupby_classes['partition'],
                countall_rule=False,
                only_fire_on_multi_key=groupby_classes['global'])
            groupby_class = groupby_classes['partition']
        else:
            raise ValueError(
                "groupby_semantics must be one of {global, partition}")

        if compiler == 'push':
            grappify_rules = grappify(
                join_type,
                self.emit_print,
                scan_array_repr, groupby_class)
        elif compiler == 'iterator':
            grappify_rules = iteratorfy(self.emit_print, scan_array_repr,
                                        groupby_class)

        # sequence that works for myrial
        rule_grps_sequence = [
            rules.remove_trivial_sequences,
            [GrappaWhileCondition()],
            rules.simple_group_by,
            cppcommon.clang_push_select,
            rules.push_project,
            rules.push_apply,
            groupby_rules,
            [rules.NumTuplesPropagation()],
            [rules.DeDupBroadcastInputs()],
            grappify_rules
        ]

        # switch join order
        if SwapJoinSides:
            rule_grps_sequence.insert(0, [rules.SwapJoinSides()])

        # set external indexing on (replacing strings with ints)
        if external_indexing:
            CBaseLanguage.set_external_indexing(True)

        # flatten the rules lists
        rule_list = list(itertools.chain(*rule_grps_sequence))

        # disable specified rules
        rules.Rule.apply_disable_flags(rule_list, *kwargs.keys())

        return rule_list
