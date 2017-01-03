# TODO: To be refactored into shared memory lang,
# where you plugin in the sequential shared memory language specific codegen

from raco import algebra
from raco import expression
from raco.backends import Algebra
from raco.backends.cpp import cppcommon
from raco import rules
from raco.pipelines import Pipelined
from raco.backends.cpp.cppcommon import StagedTupleRef, CBaseLanguage

from raco.algebra import gensym

import logging

_LOG = logging.getLogger(__name__)

import itertools


class CStagedTupleRef(StagedTupleRef):
    def __additionalDefinitionCode__(self, numfields, fieldtypes):
        constructor_template = CC.cgenv().get_template(
            'materialized_tuple_ref_additional.cpp')

        tupletypename = self.getTupleTypename()
        return constructor_template.render(locals())


class CC(CBaseLanguage):
    _template_path = 'cpp/c_templates'
    _cgenv = CBaseLanguage.__get_env_for_template_libraries__(_template_path)

    @classmethod
    def cgenv(cls):
        return cls._cgenv

    @classmethod
    def base_template(cls):
        return cls.cgenv().get_template('base_query.cpp')

    @staticmethod
    def pipeline_wrap(ident, code, attrs):

        # timing code
        if True:
            inner_code = code
            timing_template = \
                CC._cgenv.get_template('clang_pipeline_timing.cpp')

            code = timing_template.render(locals())

        return code

    @staticmethod
    def group_wrap(ident, grpcode, attrs):
        timing_template = CC._cgenv.get_template('clang_group_timing.cpp')
        inner_code = grpcode

        code = timing_template.render(locals())
        return code

    @staticmethod
    def log(txt):
        return """std::cout << "%s" << std::endl;
        """ % txt

    @staticmethod
    def log_unquoted(code, level=0):
        return """std::cout << %s << std::endl;
      """ % code

    @staticmethod
    def log_file(code, level=0):
        return """logfile << "%s" << "\\n";\n """ % code

    @staticmethod
    def log_file_unquoted(code, level=0):
        return """logfile << %s << " ";\n """ % code


class CCOperator(Pipelined, algebra.Operator):
    _language = CC

    @classmethod
    def new_tuple_ref(cls, sym, scheme):
        return CStagedTupleRef(sym, scheme)

    @classmethod
    def language(cls):
        return cls._language

    def postorder_traversal(self, func):
        return self.postorder(func)


from raco.algebra import UnaryOperator


class CMemoryScan(algebra.UnaryOperator, CCOperator):

    def produce(self, state):
        self.input.produce(state)

    # TODO: when have pipeline tree representation,
    # TODO: will have a consumeMaterialized() method instead;
    # TODO: for now we reuse the tuple-based consume
    def consume(self, inputsym, src, state):
        # now generate the scan from memory

        # TODO: generate row variable to avoid naming conflict for nested scans
        memory_scan_template = self.language().cgenv().get_template(
            'memory_scan.cpp')

        stagedTuple = state.lookupTupleDef(inputsym)
        tuple_type = stagedTuple.getTupleTypename()
        tuple_name = stagedTuple.name

        inner_plan_compiled = self.parent().consume(stagedTuple, self, state)

        code = memory_scan_template.render(locals())
        state.setPipelineProperty("type", "in_memory")
        state.addPipeline(code)
        return None

    def num_tuples(self):
        raise NotImplementedError("{}.num_tuples()".format(op=self.opname()))

    def shortStr(self):
        return "%s" % (self.opname())

    def partitioning(self):
        raise NotImplementedError()

    def __eq__(self, other):
        """
    For what we are using MemoryScan for, the only use
    of __eq__ is in hashtable lookups for CSE optimization.
    We omit self.schema because the relation_key determines
    the level of equality needed.

    @see FileScan.__eq__
    """
        return UnaryOperator.__eq__(self, other)


class CGroupBy(cppcommon.BaseCGroupby, CCOperator):
    _i = 0

    def __init__(self, *args):
        super(CGroupBy, self).__init__(*args)
        self._cgenv = cppcommon.prepend_template_relpath(
            self.language().cgenv(), '{0}/groupby'.format(CC._template_path))

    @staticmethod
    def __genHashName__():
        name = "group_hash_%03d" % CGroupBy._i
        CGroupBy._i += 1
        return name

    def produce(self, state):
        assert len(self.grouping_list) <= 2, \
            "%s does not currently support groupings of \
            more than 2 attributes" % self.__class__.__name__
        assert len(self.aggregate_list) == 1, \
            """%s currently only supports aggregates of 1 attribute
            (aggregate_list=%s)""" \
            % (self.__class__.__name__, self.aggregate_list)
        for agg_term in self.aggregate_list:
            assert isinstance(agg_term,
                              expression.BuiltinAggregateExpression), \
                """%s only supports simple aggregate expressions.
                A rule should create Apply[GroupBy]""" \
                % self.__class__.__name__

        inp_sch = self.input.scheme()
        self.useMap = len(self.grouping_list) > 0

        if self.useMap:
            if len(self.grouping_list) == 1:
                declr_template = self._cgenv.get_template(
                    '1key_declaration.cpp')
                keytype = self.language().typename(
                    self.grouping_list[0].typeof(
                        inp_sch,
                        None))
            elif len(self.grouping_list) == 2:
                declr_template = self._cgenv.get_template(
                    '2key_declaration.cpp')
                keytypes = ','.join(
                    [self.language().typename(g.typeof(inp_sch, None))
                     for g in self.grouping_list])

        else:
            initial_value = self.__get_initial_value__(
                0,
                cached_inp_sch=inp_sch)
            declr_template = self._cgenv.get_template('0key_declaration.cpp')

        valtype = self.language().typename(
            self.aggregate_list[0].typeof(
                inp_sch,
                None))

        self.hashname = CGroupBy.__genHashName__()
        hashname = self.hashname

        hash_declr = declr_template.render(locals())
        state.addDeclarations([hash_declr])

        my_sch = self.scheme()

        _LOG.debug("aggregates: %s", self.aggregate_list)
        _LOG.debug("columns: %s", self.column_list())
        _LOG.debug("groupings: %s", self.grouping_list)
        _LOG.debug("groupby scheme: %s", my_sch)
        _LOG.debug("groupby scheme[0] type: %s", type(my_sch[0]))

        self.input.produce(state)

        # now that everything is aggregated, produce the tuples
        assert (not self.useMap) \
            or isinstance(self.column_list()[0],
                          expression.AttributeRef), \
            "assumes first column is the key and " \
            "second is aggregate result: %s" % (self.column_list()[0])

        if self.useMap:
            if len(self.grouping_list) == 1:
                produce_template = self._cgenv.get_template('1key_scan.cpp')
            elif len(self.grouping_list) == 2:
                produce_template = self._cgenv.get_template('2key_scan.cpp')
        else:
            produce_template = self._cgenv.get_template('0key_scan.cpp')

        output_tuple = CStagedTupleRef(gensym(), my_sch)
        output_tuple_name = output_tuple.name
        output_tuple_type = output_tuple.getTupleTypename()
        state.addDeclarations([output_tuple.generateDefinition()])

        inner_code = self.parent().consume(output_tuple, self, state)
        code = produce_template.render(locals())
        state.setPipelineProperty("type", "in_memory")
        state.addPipeline(code)

    def consume(self, inputTuple, fromOp, state):
        if self.useMap:
            if len(self.grouping_list) == 1:
                materialize_template = self._cgenv.get_template(
                    '1key_materialize.cpp')
            elif len(self.grouping_list) == 2:
                materialize_template = self._cgenv.get_template(
                    '2key_materialize.cpp')
        else:
            materialize_template = self._cgenv.get_template(
                '0key_materialize.cpp')

        hashname = self.hashname
        tuple_name = inputTuple.name

        # make key from grouped attributes
        if self.useMap:
            inp_sch = self.input.scheme()

            key1pos = self.grouping_list[0].get_position(inp_sch)
            key1val = inputTuple.get_code(key1pos)

            if len(self.grouping_list) == 2:
                key2pos = self.grouping_list[1].get_position(inp_sch)
                key2val = inputTuple.get_code(key2pos)

        if isinstance(self.aggregate_list[0], expression.ZeroaryOperator):
            # no value needed for Zero-input aggregate,
            # but just provide the first column
            valpos = 0
        elif isinstance(self.aggregate_list[0], expression.UnaryOperator):
            # get value positions from aggregated attributes
            valpos = self.aggregate_list[0].input.get_position(self.scheme())
        else:
            assert False, "only support Unary or Zeroary aggregates"

        val = inputTuple.get_code(valpos)

        op = self.aggregate_list[0].__class__.__name__

        code = materialize_template.render(locals())
        return code


class CHashJoin(algebra.Join, CCOperator):
    _i = 0

    @staticmethod
    def __genHashName__():
        name = "hash_%03d" % CHashJoin._i
        CHashJoin._i += 1
        return name

    def __init__(self, *args):
        super(CHashJoin, self).__init__(*args)
        self._cgenv = cppcommon.prepend_template_relpath(
            self.language().cgenv(), '{0}/hashjoin'.format(CC._template_path))

    def produce(self, state):
        if not isinstance(self.condition, expression.EQ):
            msg = "The C compiler can only handle equi-join conditions of \
            a single attribute: %s" % self.condition
            raise ValueError(msg)

        left_sch = self.left.scheme()

        # find the attribute that corresponds to the right child
        self.rightCondIsRightAttr = \
            self.condition.right.position >= len(left_sch)
        self.leftCondIsRightAttr = \
            self.condition.left.position >= len(left_sch)
        assert self.rightCondIsRightAttr ^ self.leftCondIsRightAttr

        # find the attribute that corresponds to the right child
        if self.rightCondIsRightAttr:
            self.right_keypos = \
                self.condition.right.position - len(left_sch)
        else:
            self.right_keypos = \
                self.condition.left.position - len(left_sch)

        # find the attribute that corresponds to the left child
        if self.rightCondIsRightAttr:
            self.left_keypos = self.condition.left.position
        else:
            self.left_keypos = self.condition.right.position

        self.right.childtag = "right"
        # common index is defined by same right side and same key
        hashsym_and_type = state.lookupExpr((self.right, self.right_keypos))

        if not hashsym_and_type:
            # if right child never bound then store hashtable symbol and
            # call right child produce
            self._hashname = CHashJoin.__genHashName__()
            _LOG.debug("generate hashname %s for %s", self._hashname, self)
            self.right.produce(state)
        else:
            # if found a common subexpression on right child then
            # use the same hashtable
            self._hashname, self.right_type = hashsym_and_type
            _LOG.debug("reuse hash %s for %s", self._hashname, self)

        self.left.childtag = "left"
        self.left.produce(state)

    def consume(self, t, src, state):
        if src.childtag == "right":
            my_sch = self.scheme()

            declr_template = self._cgenv.get_template("hash_declaration.cpp")

            right_template = self._cgenv.get_template("insert_materialize.cpp")

            hashname = self._hashname
            keypos = self.right_keypos
            keyval = t.get_code(self.right_keypos)

            if self.rightCondIsRightAttr:
                keytype = self.language().typename(
                    self.condition.right.typeof(
                        my_sch,
                        None))
            else:
                keytype = self.language().typename(
                    self.condition.left.typeof(
                        my_sch,
                        None))

            in_tuple_type = t.getTupleTypename()
            in_tuple_name = t.name
            self.right_type = in_tuple_type

            state.saveExpr((self.right, self.right_keypos), (self._hashname,
                                                             self.right_type))

            # declaration of hash map
            hashdeclr = declr_template.render(locals())
            state.addDeclarations([hashdeclr])

            # materialization point
            code = right_template.render(locals())

            return code

        if src.childtag == "left":
            left_template = self._cgenv.get_template("lookup.cpp")

            hashname = self._hashname
            keyname = t.name
            keytype = t.getTupleTypename()
            keypos = self.left_keypos
            keyval = t.get_code(keypos)

            right_tuple_name = gensym()

            outTuple = CStagedTupleRef(gensym(), self.scheme())
            out_tuple_type_def = outTuple.generateDefinition()
            out_tuple_type = outTuple.getTupleTypename()
            out_tuple_name = outTuple.name

            type1 = keytype
            type1numfields = len(t.scheme)
            type2 = self.right_type
            type2numfields = len(self.right.scheme())
            append_func_name, combine_function_def = \
                CStagedTupleRef.get_append(
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


# iteration  over table + insertion into hash table with filter

class CUnionAll(cppcommon.CBaseUnionAll, CCOperator):
    pass


class CApply(cppcommon.CBaseApply, CCOperator):
    pass


class CProject(cppcommon.CBaseProject, CCOperator):
    pass


class CSelect(cppcommon.CBaseSelect, CCOperator):
    pass


class CFileScan(cppcommon.CBaseFileScan, CCOperator):

    def __get_ascii_scan_template__(self):
        return CC.cgenv().get_template('ascii_scan.cpp')

    def __get_binary_scan_template__(self):
        # TODO binary input
        return CC.cgenv().get_template('ascii_scan.cpp')

    def __get_relation_decl_template__(self, name):
        return CC.cgenv().get_template('relation_declaration.cpp')


class CSink(cppcommon.CBaseSink, CCOperator):
    pass


class CStore(cppcommon.CBaseStore, CCOperator):

    def __file_code__(self, t, state):
        output_stream_symbol = "outputfile"
        count_symbol = "_result_count"
        filename = str(self.relation_key).split(":")[2]
        count_filename = filename + ".count"
        count_decl = \
            CC.cgenv().get_template("groupby/0key_declaration.cpp").render(
                valtype="uint64_t",
                hashname=count_symbol,
                initial_value=0)
        stream_decl = CC.cgenv().get_template("output_stream_decl.cpp").render(
            output_stream_symbol=output_stream_symbol)
        stream_open = CC.cgenv().get_template(
            "output_stream_open.cpp").render(locals())
        scheme_write = self.__write_schema(self.scheme())
        state.addInitializers(
            [count_decl, stream_decl, stream_open, scheme_write])

        code = "{0}.toOStreamAscii({1});\n".format(
            t.name,
            output_stream_symbol)
        code += "{0}++;\n".format(count_symbol)

        stream_close = \
            CC.cgenv().get_template("output_stream_close.cpp").render(
                output_stream_symbol=output_stream_symbol)
        write_count = CC.cgenv().get_template("write_count.cpp").render(
            filename=count_filename,
            count_symbol=count_symbol)
        state.addCleanups([stream_close, write_count])

        return code

    def __write_schema(self, scheme):
        output_stream_symbol = "out_scheme_file"
        schemafile = str(self.relation_key).split(":")[2] + ".scheme"
        code = CC.cgenv().get_template("output_stream_decl.cpp").render(
            output_stream_symbol=output_stream_symbol)
        code += CC.cgenv().get_template("output_stream_open.cpp").render(
            output_stream_symbol=output_stream_symbol,
            filename=schemafile)
        names = [x.encode('UTF8') for x in scheme.get_names()]

        code += CC.cgenv().get_template("output_stream_write.cpp").render(
            output_stream_symbol=output_stream_symbol,
            stringval="{0}".format(names))
        code += CC.cgenv().get_template("output_stream_write.cpp").render(
            output_stream_symbol=output_stream_symbol,
            stringval="{0}".format(
                scheme.get_types()))

        code += CC.cgenv().get_template("output_stream_close.cpp").render(
            output_stream_symbol=output_stream_symbol)
        return code


class MemoryScanOfFileScan(rules.Rule):

    """A rewrite rule for making a scan into
    materialization in memory then memory scan"""

    def fire(self, expr):
        if isinstance(expr, algebra.Scan) and not isinstance(expr, CFileScan):
            return CMemoryScan(CFileScan(expr.relation_key, expr.scheme()))
        return expr

    def __str__(self):
        return "Scan => MemoryScan[FileScan]"


def clangify(emit_print):
    return [
        rules.ProjectingJoinToProjectOfJoin(),

        rules.OneToOne(algebra.Select, CSelect),
        MemoryScanOfFileScan(),
        rules.OneToOne(algebra.Apply, CApply),
        rules.OneToOne(algebra.Join, CHashJoin),
        rules.OneToOne(algebra.GroupBy, CGroupBy),
        rules.OneToOne(algebra.Project, CProject),
        rules.OneToOne(algebra.UnionAll, CUnionAll),
        cppcommon.StoreToBaseCStore(emit_print, CStore),
        rules.OneToOne(algebra.Sink, CSink),

        cppcommon.BreakHashJoinConjunction(CSelect, CHashJoin)
    ]


class CCAlgebra(Algebra):

    def __init__(self, emit_print=cppcommon.EMIT_CONSOLE):
        """ To store results into a file or onto console """
        self.emit_print = emit_print

    def opt_rules(self, **kwargs):
        # Sequence that works for datalog
        # TODO: replace with below
        # datalog_rules = [
        # rules.CrossProduct2Join(),
        # rules.SimpleGroupBy(),
        # rules.OneToOne(algebra.Select, CSelect),
        # MemoryScanOfFileScan(),
        # rules.OneToOne(algebra.Apply, CApply),
        # rules.OneToOne(algebra.Join, CHashJoin),
        # rules.OneToOne(algebra.GroupBy, CGroupBy),
        # rules.OneToOne(algebra.Project, CProject),
        # TODO: obviously breaks semantics
        # rules.OneToOne(algebra.Union, CUnionAll),
        # rules.FreeMemory()
        # ]

        # sequence that works for myrial
        rule_grps_sequence = [
            rules.remove_trivial_sequences,
            rules.simple_group_by,
            cppcommon.clang_push_select,
            [rules.ProjectToDistinctColumnSelect(),
             rules.JoinToProjectingJoin()],
            rules.push_apply,
            [rules.DeDupBroadcastInputs()],
            clangify(self.emit_print)
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
