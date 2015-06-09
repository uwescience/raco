
# TODO: To be refactored into parallel shared memory lang,
# where you plugin in the parallel shared memory language specific codegen

from raco import algebra
from raco.expression import aggregate
from raco import expression
from raco.language import Algebra
from raco import rules
from raco.pipelines import Pipelined
from raco.language.clangcommon import StagedTupleRef, CBaseLanguage
from raco.language import clangcommon
from raco.utility import emitlist
from raco import types

from raco.algebra import gensym

import logging
_LOG = logging.getLogger(__name__)

import itertools


def define_cl_arg(type, name, default_value, description):
    return GrappaLanguage.cgenv().get_template(
        'define_cl_arg.cpp').render(locals())


class GrappaStagedTupleRef(StagedTupleRef):
    def __afterDefinitionCode__(self, numfields, fieldtypes):
        # Grappa requires structures to be block aligned if they will be
        # iterated over with localizing forall
        return "GRAPPA_BLOCK_ALIGNED"


class GrappaLanguage(CBaseLanguage):
    _template_path = 'grappa_templates'
    _cgenv = CBaseLanguage.__get_env_for_template_libraries__(_template_path)

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

        timing_template = GrappaLanguage.cgenv().get_template(
            'grappa_pipeline_timing.cpp')
        wrappers.append((timing_template, locals()))

        dependences = attrs.get('dependences', set())
        assert isinstance(dependences, set)
        _LOG.debug("pipeline %s dependences %s", ident, dependences)

        dependence_code = emitlist([wait_statement(d) for d in dependences])
        dependence_template = GrappaLanguage.cgenv().from_string("""
        {{dependence_code}}
        {{inner_code}}
        """)
        wrappers.append((dependence_template, locals()))

        syncname = attrs.get('sync')
        if syncname:
            dependence_captures = emitlist(
                [",&{dep}".format(dep=d) for d in dependences])
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


# TODO: replace with ScanTemp functionality?
class GrappaMemoryScan(algebra.UnaryOperator, GrappaOperator):

    def num_tuples(self):
        return 10000  # placeholder

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

        memory_scan_template = self.language().cgenv().get_template(
            'memory_scan.cpp')

        stagedTuple = state.lookupTupleDef(inputsym)
        tuple_type = stagedTuple.getTupleTypename()
        tuple_name = stagedTuple.name

        inner_code = self.parent().consume(stagedTuple, self, state)

        code = memory_scan_template.render(locals())
        state.setPipelineProperty('type', 'in_memory')
        state.setPipelineProperty('source', self.__class__)
        state.addPipeline(code)
        return None

    def shortStr(self):
        return "%s" % (self.opname())

    def __eq__(self, other):
        """
        See important __eq__ notes below
        @see FileScan.__eq__
        """
        return UnaryOperator.__eq__(self, other)


class GrappaJoin(algebra.Join, GrappaOperator):

    @classmethod
    def __aggregate_val__(cls, tuple, cols):
        return "std::make_tuple({0})".format(
            ','.join([tuple.get_code(p) for p in cols]))

    @classmethod
    def __aggregate_type__(cls, sch, cols):
        return "std::tuple<{0}>".format(
            ','.join([cls.language().typename(
                expression.UnnamedAttributeRef(c).typeof(sch, None))
                for c in cols]))


class GrappaSymmetricHashJoin(GrappaJoin, GrappaOperator):
    _i = 0

    @classmethod
    def __genBaseName__(cls):
        name = "%03d" % cls._i
        cls._i += 1
        return name

    def __getHashName__(self):
        name = "dhash_%s" % self.symBase
        return name

    def __init__(self, *args):
        super(GrappaSymmetricHashJoin, self).__init__(*args)
        self._cgenv = clangcommon.prepend_template_relpath(
            self.language().cgenv(),
            '{0}/symmetrichashjoin'.format(GrappaLanguage._template_path))

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
        keytype = self.__aggregate_type__(my_sch, self.rightcols)
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
        state.addInitializers([init_template.render(locals())])
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

        if src.childtag == "right":
            left_sch = self.left.scheme()

            # save for later
            self.right_in_tuple_type = t.getTupleTypename()
            state.resolveSymbol(self.rightTypeRef, self.right_in_tuple_type)

            inner_plan_compiled = self.parent().consume(outTuple, self, state)

            keyval = self.__aggregate_val__(t, self.rightcols)

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

            # need to add later because requires left tuple type decl
            self.right_combine_decl = combine_function_def

            code = access_template.render(locals())
            return code

        if src.childtag == "left":
            right_in_tuple_type = self.right_in_tuple_type
            left_in_tuple_type = t.getTupleTypename()
            state.resolveSymbol(self.leftTypeRef, left_in_tuple_type)

            keyval = self.__aggregate_val__(t, self.leftcols)

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

            state.addDeclarations([self.right_combine_decl,
                                   combine_function_def])

            code = access_template.render(locals())
            return code

        assert False, "src not equal to left or right"


class GrappaShuffleHashJoin(algebra.Join, GrappaOperator):
    _i = 0

    @classmethod
    def __genBaseName__(cls):
        name = "%03d" % cls._i
        cls._i += 1
        return name

    def __getHashName__(self):
        name = "hashjoin_reducer_%s" % self.symBase
        return name

    def __init__(self, *args):
        super(GrappaShuffleHashJoin, self).__init__(*args)
        self._cgenv = clangcommon.prepend_template_relpath(
            self.language().cgenv(),
            '{0}/shufflehashjoin'.format(GrappaLanguage._template_path))

    def produce(self, state):
        left_sch = self.left.scheme()

        self.syncnames = []
        self.symBase = self.__genBaseName__()

        self.right.childtag = "right"
        self.rightTupleTypeRef = None  # may remain None if CSE succeeds
        self.leftTupleTypeRef = None  # may remain None if CSE succeeds

        # find the attribute that corresponds to the right child
        self.rightCondIsRightAttr = \
            self.condition.right.position >= len(left_sch)
        self.leftCondIsRightAttr = \
            self.condition.left.position >= len(left_sch)
        assert self.rightCondIsRightAttr ^ self.leftCondIsRightAttr

        # find right key position
        if self.rightCondIsRightAttr:
            self.right_keypos = self.condition.right.position \
                - len(left_sch)
        else:
            self.right_keypos = self.condition.left.position \
                - len(left_sch)

        # find left key position
        if self.rightCondIsRightAttr:
            self.left_keypos = self.condition.left.position
        else:
            self.left_keypos = self.condition.right.position

        # define output tuple
        outTuple = GrappaStagedTupleRef(gensym(), self.scheme())
        out_tuple_type_def = outTuple.generateDefinition()
        out_tuple_type = outTuple.getTupleTypename()
        out_tuple_name = outTuple.name

        # common index is defined by same right side and same key
        # TODO: probably want also left side
        hashtableInfo = state.lookupExpr((self.right, self.right_keypos))
        if not hashtableInfo:
            # if right child never bound then store hashtable symbol and
            # call right child produce
            self._hashname = self.__getHashName__()
            _LOG.debug("generate hashname %s for %s", self._hashname, self)

            hashname = self._hashname

            # declaration of hash map
            self.rightTupleTypeRef = state.createUnresolvedSymbol()
            self.leftTupleTypeRef = state.createUnresolvedSymbol()
            self.outTupleTypeRef = state.createUnresolvedSymbol()
            right_type = self.rightTupleTypeRef.getPlaceholder()
            left_type = self.leftTupleTypeRef.getPlaceholder()

            # TODO: really want this addInitializers to be addPreCode
            # TODO: *for all pipelines that use this hashname*
            init_template = self._cgenv.get_template('hash_init.cpp')

            state.addInitializers([init_template.render(locals())])
            self.right.produce(state)

            self.left.childtag = "left"
            self.left.produce(state)

            state.saveExpr((self.right, self.right_keypos),
                           (self._hashname, right_type, left_type,
                            self.right_syncname, self.left_syncname))

        else:
            # if found a common subexpression on right child then
            # use the same hashtable
            self._hashname, right_type, left_type,\
                self.right_syncname, self.left_syncname = hashtableInfo
            _LOG.debug("reuse hash %s for %s", self._hashname, self)

        # now that Relation is produced, produce its contents by iterating over
        # the join result
        iterate_template = self._cgenv.get_template('result_scan.cpp')

        hashname = self._hashname

        state.addDeclarations([out_tuple_type_def])

        pipeline_sync = create_pipeline_synchronization(state)
        get_pipeline_task_name(state)

        # add dependences on left and right inputs
        state.addToPipelinePropertySet('dependences', self.right_syncname)
        state.addToPipelinePropertySet('dependences', self.left_syncname)

        # reduce is a single self contained pipeline.
        # future hashjoin implementations may pipeline out of it
        # by passing a continuation to reduceExecute
        reduce_template = self._cgenv.get_template('reduce.cpp')

        state.addPreCode(reduce_template.render(locals()))

        delete_template = self._cgenv.get_template('delete.cpp')

        state.addPostCode(delete_template.render(locals()))

        inner_code_compiled = self.parent().consume(outTuple, self, state)

        code = iterate_template % locals()
        state.setPipelineProperty('type', 'in_memory')
        state.setPipelineProperty('source', self.__class__)
        state.addPipeline(code)

    def consume(self, inputTuple, fromOp, state):
        if fromOp.childtag == "right":
            side = "Right"
            self.right_syncname = get_pipeline_task_name(state)

            keypos = self.right_keypos

            self.rightTupleTypename = inputTuple.getTupleTypename()
            if self.rightTupleTypeRef is not None:
                state.resolveSymbol(self.rightTupleTypeRef,
                                    self.rightTupleTypename)
        elif fromOp.childtag == "left":
            side = "Left"
            self.left_syncname = get_pipeline_task_name(state)

            keypos = self.left_keypos

            self.leftTupleTypename = inputTuple.getTupleTypename()
            if self.leftTupleTypeRef is not None:
                state.resolveSymbol(self.leftTupleTypeRef,
                                    self.leftTupleTypename)
        else:
            assert False, "src not equal to left or right"

        hashname = self._hashname
        keyname = inputTuple.name
        keytype = inputTuple.getTupleTypename()
        keyval = inputTuple.get_code(keypos)

        # intra-pipeline sync
        global_syncname = state.getPipelineProperty('global_syncname')

        mat_template = self._cgenv.get_template('materialize.cpp')

        # materialization point
        code = mat_template.render(locals())
        return code


class GrappaGroupBy(clangcommon.BaseCGroupby, GrappaOperator):
    _i = 0

    _ONE_BUILT_IN = 0
    _MULTI_UDA = 1

    @classmethod
    def __genHashName__(cls):
        name = "group_hash_%03d" % cls._i
        cls._i += 1
        return name

    def __init__(self, *args):
        super(GrappaGroupBy, self).__init__(*args)
        self._cgenv = clangcommon.prepend_template_relpath(
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
            aggregate.MAX: 'std::numeric_limits<{0}>::lowest',
            aggregate.MIN: 'std::numeric_limits<{0}>::max'
        }.get(op.__class__)
        if r is None:
            return 'Aggregates::Zero'
        else:
            return r

    def produce(self, state):
        self._agg_mode = None
        if len(self.aggregate_list) == 1 \
                and isinstance(self.aggregate_list[0],
                               expression.BuiltinAggregateExpression):
            self._agg_mode = self._ONE_BUILT_IN
        elif all([isinstance(a, expression.UdaAggregateExpression)
                  for a in self.aggregate_list]):
            self._agg_mode = self._MULTI_UDA

        assert self._agg_mode is not None, \
            "unsupported aggregates {0}".format(self.aggregate_list)
        _LOG.debug("%s _agg_mode was set to %s", self, self._agg_mode)

        self.useKey = len(self.grouping_list) > 0
        _LOG.debug("groupby uses keys? %s" % self.useKey)

        inp_sch = self.input.scheme()

        if self._agg_mode == self._ONE_BUILT_IN:
            state_type = self.language().typename(
                self.aggregate_list[0].input.typeof(inp_sch, None))
            op = self.aggregate_list[0]
            up_op_name = op.__class__.__name__
            co_op_name = self._combiner_for_builtin_update(
                op).__class__.__name__
            self.update_func = "Aggregates::{op}<{type}, {type}>".format(
                op=up_op_name, type=state_type)
            combine_func = "Aggregates::{op}<{type}, {type}>".format(
                op=co_op_name, type=state_type)
        elif self._agg_mode == self._MULTI_UDA:
            # for now just name the aggregate after the first state variable
            self.func_name = self.updaters[0][0]
            self.state_tuple = GrappaStagedTupleRef(gensym(),
                                                    self.state_scheme)
            state.addDeclarations([self.state_tuple.generateDefinition()])
            state_type = self.state_tuple.getTupleTypename()
            self.update_func = "{name}_update".format(name=self.func_name)

        update_func = self.update_func

        if self.useKey:
            numkeys = len(self.grouping_list)
            keytype = "std::tuple<{types}>".format(
                types=','.join([self.language().typename(
                    g.typeof(inp_sch, None)) for g in self.grouping_list]))

        self._hashname = self.__genHashName__()
        _LOG.debug("generate hashname %s for %s", self._hashname, self)

        hashname = self._hashname

        if self.useKey:
            init_template = self._cgenv.get_template('withkey_init.cpp')
            valtype = state_type
        else:
            if self._agg_mode == self._ONE_BUILT_IN:
                initial_value = \
                    self.__get_initial_value__(0, cached_inp_sch=inp_sch)
                no_key_state_initializer = \
                    "counter<{state_type}>::create({valinit})".format(
                        state_type=state_type, valinit=initial_value)
            elif self._agg_mode == self._MULTI_UDA:
                no_key_state_initializer = \
                    "symmetric_global_alloc<{state_tuple_type}>()".format(
                        state_tuple_type=self.state_tuple.getTupleTypename())

            init_template = self._cgenv.get_template('withoutkey_init.cpp')
            initializer = no_key_state_initializer

        state.addInitializers([init_template.render(locals())])

        self.input.produce(state)

        # now that everything is aggregated, produce the tuples
        # assert len(self.column_list()) == 1 \
        #    or isinstance(self.column_list()[0],
        #                  expression.AttributeRef), \
        #    """assumes first column is the key and second is aggregate result
#            column_list: %s""" % self.column_list()

        if self.useKey:
            mapping_var_name = gensym()
            if self._agg_mode == self._ONE_BUILT_IN:
                emit_type = self.language().typename(
                    self.aggregate_list[0].input.typeof(
                        self.input.scheme(), None))
            elif self._agg_mode == self._MULTI_UDA:
                emit_type = self.state_tuple.getTupleTypename()

            if self._agg_mode == self._ONE_BUILT_IN:
                # need to force type in make_tuple
                produce_template = self._cgenv.get_template(
                    'one_built_in_scan.cpp')
            elif self._agg_mode == self._MULTI_UDA:
                # pass in attribute values individually
                produce_template = self._cgenv.get_template(
                    'multi_uda_scan.cpp')

        else:
            if self._agg_mode == self._ONE_BUILT_IN:
                produce_template = self._cgenv.get_template(
                    'one_built_in_0key_output.cpp')

            elif self._agg_mode == self._MULTI_UDA:
                produce_template = self._cgenv.get_template(
                    'multi_uda_0key_output.cpp')

        pipeline_sync = create_pipeline_synchronization(state)
        get_pipeline_task_name(state)

        # add a dependence on the input aggregation pipeline
        state.addToPipelinePropertySet('dependences', self.input_syncname)

        output_tuple = GrappaStagedTupleRef(gensym(), self.scheme())
        output_tuple_name = output_tuple.name
        output_tuple_type = output_tuple.getTupleTypename()
        output_tuple_set_func = output_tuple.set_func_code(0)
        state.addDeclarations([output_tuple.generateDefinition()])

        inner_code = self.parent().consume(output_tuple, self, state)
        code = produce_template.render(locals())
        state.setPipelineProperty("type", "in_memory")
        state.addPipeline(code)

    def consume(self, inputTuple, fromOp, state):
        # save the inter-pipeline task name
        self.input_syncname = get_pipeline_task_name(state)

        inp_sch = self.input.scheme()

        all_decls = []
        all_inits = []

        # compile update statements
        def compile_assignments(assgns):
            state_var_update_template = "auto {assignment};"
            state_var_updates = []
            state_vars = []
            decls = []
            inits = []

            for a in assgns:
                state_name, update_exp = a
                # doesn't have to use inputTuple.name,
                # but it will for simplicity
                rhs = self.language().compile_expression(
                    update_exp,
                    tupleref=inputTuple,
                    state_scheme=self.state_scheme)

                # combine lhs, rhs with assignment
                code = "{lhs} = {rhs}".format(lhs=state_name, rhs=rhs[0])

                decls += rhs[1]
                inits += rhs[2]

                state_var_updates.append(
                    state_var_update_template.format(assignment=code))
                state_vars.append(state_name)

            return state_var_updates, state_vars, decls, inits

        update_updates, update_state_vars, update_decls, update_inits = \
            compile_assignments(self.updaters)
        init_updates, init_state_vars, init_decls, init_inits = \
            compile_assignments(self.inits)
        assert set(update_state_vars) == set(init_state_vars), \
            """Initialized and update state vars are not the same \
            (may not need to be?)"""
        all_decls += update_decls + init_decls
        all_inits += update_inits + init_inits

        if self._agg_mode == self._MULTI_UDA:
            state_tuple_decl = self.state_tuple.generateDefinition()
            update_def = self._cgenv.get_template(
                'update_definition.cpp').render(
                    state_type=self.state_tuple.getTupleTypename(),
                    input_type=inputTuple.getTupleTypename(),
                    input_tuple_name=inputTuple.name,
                    update_updates=update_updates,
                    update_state_vars=update_state_vars,
                    name=self.func_name)
            init_def = self._cgenv.get_template('init_definition.cpp').render(
                state_type=self.state_tuple.getTupleTypename(),
                init_updates=init_updates,
                init_state_vars=init_state_vars,
                name=self.func_name)

            all_decls += [update_def, init_def]

        # form code to fill in the materialize template
        if self._agg_mode == self._ONE_BUILT_IN:

            if isinstance(self.aggregate_list[0], expression.ZeroaryOperator):
                # no value needed for Zero-input aggregate,
                # but just provide the first column
                valpos = 0
            elif isinstance(self.aggregate_list[0], expression.UnaryOperator):
                # get value positions from aggregated attributes
                valpos = \
                    self.aggregate_list[0].input.get_position(self.scheme())
            else:
                assert False, "only support Unary or Zeroary aggregates"

            update_val = inputTuple.get_code(valpos)
            input_type = self.language().typename(
                self.aggregate_list[0].input.typeof(inp_sch, None))

            init_func = self._init_func_for_op(self.aggregate_list[0])\
                .format(input_type)

        elif self._agg_mode == self._MULTI_UDA:
            init_func = "{name}_init".format(name=self.func_name)
            update_val = inputTuple.name
            input_type = inputTuple.getTupleTypename()

        if self.useKey:
            numkeys = len(self.grouping_list)
            keygets = [inputTuple.get_code(g.get_position(inp_sch))
                       for g in self.grouping_list]

            materialize_template = self._cgenv.get_template('nkey_update.cpp')
        else:
            if self._agg_mode == self._ONE_BUILT_IN:
                materialize_template = self._cgenv.get_template(
                    'one_built_in_0key_update.cpp')
            elif self._agg_mode == self._MULTI_UDA:
                materialize_template = self._cgenv.get_template(
                    'multi_uda_0key_update.cpp')

        hashname = self._hashname
        tuple_name = inputTuple.name
        pipeline_sync = state.getPipelineProperty("global_syncname")

        state.addDeclarations(all_decls)
        state.addInitializers(all_inits)

        update_func = self.update_func

        code = materialize_template.render(locals())
        return code


def wait_statement(name):
    return GrappaLanguage.cgenv().get_template(
        'wait_statement.cpp').render(name=name)


def get_pipeline_task_name(state):
    name = "p_task_{n}".format(n=state.getCurrentPipelineId())
    state.setPipelineProperty('sync', name)
    wait_stmt = wait_statement(name)
    state.addMainWaitStatement(wait_stmt)
    return name


class GrappaHashJoin(GrappaJoin, GrappaOperator):
    _i = 0

    @classmethod
    def __genHashName__(cls):
        name = "hash_%03d" % cls._i
        cls._i += 1
        return name

    def __init__(self, *args):
        super(GrappaHashJoin, self).__init__(*args)
        self._cgenv = clangcommon.prepend_template_relpath(
            self.language().cgenv(),
            '{0}/hashjoin'.format(GrappaLanguage._template_path))

    def produce(self, state):
        declr_template = self._cgenv.get_template('hash_declaration.cpp')

        self.right.childtag = "right"
        self.rightTupleTypeRef = None  # may remain None if CSE succeeds

        my_sch = self.scheme()
        left_sch = self.left.scheme()
        right_sch = self.right.scheme()

        self.leftcols, self.rightcols = \
            algebra.convertcondition(self.condition,
                                     len(left_sch),
                                     left_sch + right_sch)

        keytype = self.__aggregate_type__(my_sch, self.rightcols)

        # common index is defined by same right side and same key
        hashtableInfo = state.lookupExpr((self.right,
                                          frozenset(self.rightcols)))
        if not hashtableInfo:
            # if right child never bound then store hashtable symbol and
            # call right child produce
            self._hashname = self.__genHashName__()
            _LOG.debug("generate hashname %s for %s", self._hashname, self)

            hashname = self._hashname

            # declaration of hash map
            self.rightTupleTypeRef = state.createUnresolvedSymbol()
            in_tuple_type = self.rightTupleTypeRef.getPlaceholder()
            hashdeclr = declr_template.render(locals())
            state.addDeclarationsUnresolved([hashdeclr])

            init_template = self._cgenv.get_template('hash_init.cpp')

            state.addInitializers([init_template.render(locals())])
            self.right.produce(state)
            state.saveExpr((self.right, frozenset(self.rightcols)),
                           (self._hashname, self.rightTupleTypename,
                            self.right_syncname))
            # TODO always safe here? I really want to call
            # TODO saveExpr before self.right.produce(),
            # TODO but I need to get the self.rightTupleTypename cleanly
        else:
            # if found a common subexpression on right child then
            # use the same hashtable
            self._hashname, self.rightTupleTypename, self.right_syncname\
                = hashtableInfo
            _LOG.debug("reuse hash %s for %s", self._hashname, self)

        self.left.childtag = "left"
        self.left.produce(state)

    def consume(self, t, src, state):
        if src.childtag == "right":
            right_template = self._cgenv.get_template('insert_materialize.cpp')

            hashname = self._hashname
            keyname = t.name
            keyval = self.__aggregate_val__(t, self.rightcols)

            self.right_syncname = get_pipeline_task_name(state)

            self.rightTupleTypename = t.getTupleTypename()
            if self.rightTupleTypeRef is not None:
                state.resolveSymbol(self.rightTupleTypeRef,
                                    self.rightTupleTypename)

            pipeline_sync = state.getPipelineProperty('global_syncname')

            # materialization point
            code = right_template.render(locals())

            return code

        if src.childtag == "left":
            left_template = self._cgenv.get_template('lookup.cpp')

            # add a dependence on the right pipeline
            state.addToPipelinePropertySet('dependences', self.right_syncname)

            hashname = self._hashname
            keyname = t.name
            input_tuple_type = t.getTupleTypename()
            keyval = self.__aggregate_val__(t, self.leftcols)

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
class GrappaSelect(clangcommon.CBaseSelect, GrappaOperator):
    pass


# Basic apply like serial C++
class GrappaApply(clangcommon.CBaseApply, GrappaOperator):
    pass


# Basic duplication based bag union like serial C++
class GrappaUnionAll(clangcommon.CBaseUnionAll, GrappaOperator):
    pass


# Basic materialized copy based project like serial C++
class GrappaProject(clangcommon.CBaseProject, GrappaOperator):
    pass


class GrappaFileScan(clangcommon.CBaseFileScan, GrappaOperator):

    def __get_ascii_scan_template__(self):
        _LOG.warn("binary/ascii is command line choice")
        return self._language.cgenv().get_template('file_scan.cpp')

    def __get_binary_scan_template__(self):
        _LOG.warn("binary/ascii is command line choice")
        return self._language.cgenv().get_template('file_scan.cpp')

    def __get_relation_decl_template__(self, name):
        return self._language.cgenv().get_template('relation_declaration.cpp')


class GrappaStore(clangcommon.CBaseStore, GrappaOperator):

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


class MemoryScanOfFileScan(rules.Rule):

    """A rewrite rule for making a scan into materialization
     in memory then memory scan"""

    def fire(self, expr):
        if isinstance(expr, algebra.Scan) \
                and not isinstance(expr, GrappaFileScan):
            return GrappaMemoryScan(GrappaFileScan(expr.relation_key,
                                                   expr.scheme()))
        return expr

    def __str__(self):
        return "Scan => MemoryScan(FileScan)"


def grappify(join_type, emit_print):
    return [
        rules.ProjectingJoinToProjectOfJoin(),

        rules.OneToOne(algebra.Select, GrappaSelect),
        MemoryScanOfFileScan(),
        rules.OneToOne(algebra.Apply, GrappaApply),
        rules.OneToOne(algebra.Join, join_type),
        rules.OneToOne(algebra.GroupBy, GrappaGroupBy),
        rules.OneToOne(algebra.Project, GrappaProject),
        rules.OneToOne(algebra.UnionAll, GrappaUnionAll),
        # TODO: obviously breaks semantics
        rules.OneToOne(algebra.Union, GrappaUnionAll),
        clangcommon.StoreToBaseCStore(emit_print, GrappaStore),

        # Don't need this because we support two-key
        # clangcommon.BreakHashJoinConjunction(GrappaSelect, join_type)
    ]


class GrappaAlgebra(Algebra):

    def __init__(self, emit_print=clangcommon.EMIT_CONSOLE):
        self.emit_print = emit_print

    def opt_rules(self, **kwargs):
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

        join_type = kwargs.get('join_type', GrappaHashJoin)

        # sequence that works for myrial
        rule_grps_sequence = [
            rules.remove_trivial_sequences,
            rules.simple_group_by,
            clangcommon.clang_push_select,
            rules.push_project,
            rules.push_apply,
            grappify(join_type, self.emit_print)
        ]

        if kwargs.get('SwapJoinSides'):
            rule_grps_sequence.insert(0, [rules.SwapJoinSides()])

        # set external indexing on (replacing strings with ints)
        if kwargs.get('external_indexing'):
            CBaseLanguage.set_external_indexing(True)

        # flatten the rules lists
        rule_list = list(itertools.chain(*rule_grps_sequence))

        # disable specified rules
        rules.Rule.apply_disable_flags(rule_list, *kwargs.keys())

        return rule_list
