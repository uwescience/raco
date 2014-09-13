# TODO: To be refactored into shared memory lang,
# where you plugin in the sequential shared memory language specific codegen

from raco import algebra
from raco import expression
from raco.language import clangcommon, Algebra
from raco import rules
from raco.pipelines import Pipelined
from raco.language.clangcommon import StagedTupleRef, ct, CBaseLanguage

from raco.algebra import gensym

import logging

_LOG = logging.getLogger(__name__)

import itertools
import os.path


template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "c_templates")


def readtemplate(fname):
    return file(os.path.join(template_path, fname)).read()


template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "c_templates")

base_template = readtemplate("base_query.template")
twopass_select_template = readtemplate("precount_select.template")
hashjoin_template = readtemplate("hashjoin.template")
filteringhashjoin_template = ""
filtering_nestedloop_join_chain_template = ""
# =readtemplate("filtering_nestedloop_join_chain.template")
ascii_scan_template = readtemplate("ascii_scan.template")
binary_scan_template = readtemplate("binary_scan.template")


class CStagedTupleRef(StagedTupleRef):
    def __additionalDefinitionCode__(self):
        constructor_template = """
    public:
    %(tupletypename)s (relationInfo * rel, int row) {
      %(copies)s
    }
    """

        copytemplate = """_fields[%(fieldnum)s] = \
        rel->relation[row*rel->fields + %(fieldnum)s];
    """

        copies = ""
        # TODO: actually list the trimmed schema offsets
        for i in range(0, len(self.scheme)):
            fieldnum = i
            copies += copytemplate % locals()

        tupletypename = self.getTupleTypename()
        return constructor_template % locals()


class CC(CBaseLanguage):
    @staticmethod
    def base_template():
        return base_template

    @staticmethod
    def pipeline_wrap(ident, code, attrs):

        # timing code
        if True:
            inner_code = code
            timing_template = ct("""auto start_%(ident)s = walltime();
            %(inner_code)s
            auto end_%(ident)s = walltime();
            auto runtime_%(ident)s = end_%(ident)s - start_%(ident)s;
            std::cout << "pipeline %(ident)s: " << runtime_%(ident)s \
                        << " s" \
                        << std::endl;
            std::cout << "timestamp %(ident)s start " \
                        << std::setprecision(15) \
                        << start_%(ident)s << std::endl;
            std::cout << "timestamp %(ident)s end " \
                        << std::setprecision(15) \
                        << end_%(ident)s << std::endl;
            """)
            code = timing_template % locals()

        return code

    @staticmethod
    def group_wrap(ident, grpcode, attrs):
        pipeline_template = ct("""
        auto start_%(ident)s = walltime();
        %(grpcode)s
        auto end_%(ident)s = walltime();
        auto runtime_%(ident)s = end_%(ident)s - start_%(ident)s;
        std::cout << "pipeline group %(ident)s: " \
                    << runtime_%(ident)s \
                    << " s" << std::endl;
        """)

        code = pipeline_template % locals()
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

    @classmethod
    def compile_stringliteral(cls, s):
        sid = cls.newstringident()
        lookup_init = """auto %s = string_index.string_lookup(%s);""" \
                      % (sid, s)
        build_init = """
        string_index = build_string_index("sp2bench_1m.index");
        """
        return """(%s)""" % sid, [], [build_init, lookup_init]
        # raise ValueError("String Literals not supported\
        # in C language: %s" % s)


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
        memory_scan_template = """for (uint64_t i : %(inputsym)s->range()) {
          %(tuple_type)s %(tuple_name)s(%(inputsym)s, i);

          %(inner_plan_compiled)s
       } // end scan over %(inputsym)s
       """

        stagedTuple = state.lookupTupleDef(inputsym)
        tuple_type = stagedTuple.getTupleTypename()
        tuple_name = stagedTuple.name

        inner_plan_compiled = self.parent().consume(stagedTuple, self, state)

        code = memory_scan_template % locals()
        state.setPipelineProperty("type", "in_memory")
        state.addPipeline(code)
        return None

    def num_tuples(self):
        raise NotImplementedError("{}.num_tuples()".format(op=self.opname()))

    def shortStr(self):
        return "%s" % (self.opname())

    def __eq__(self, other):
        """
    For what we are using MemoryScan for, the only use
    of __eq__ is in hashtable lookups for CSE optimization.
    We omit self.schema because the relation_key determines
    the level of equality needed.

    @see FileScan.__eq__
    """
        return UnaryOperator.__eq__(self, other)


class CGroupBy(algebra.GroupBy, CCOperator):
    _i = 0

    @classmethod
    def __genHashName__(cls):
        name = "group_hash_%03d" % cls._i
        cls._i += 1
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

        self.useMap = len(self.grouping_list) > 0

        if self.useMap:
            if len(self.grouping_list) == 1:
                declr_template = """std::unordered_map<int64_t, int64_t> \
                %(hashname)s;
          """
            elif len(self.grouping_list) == 2:
                declr_template = """std::unordered_map<\
                std::pair<int64_t, int64_t>, int64_t, pairhash> \
                %(hashname)s;
                """
        else:
            declr_template = """int64_t %(hashname)s;
            """

        self.hashname = self.__genHashName__()
        hashname = self.hashname

        hash_declr = declr_template % locals()
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
                produce_template = """for (auto it=%(hashname)s.begin(); \
                it!=%(hashname)s.end(); it++) {
                %(output_tuple_type)s %(output_tuple_name)s(\
                {it->first, it->second});
                %(inner_code)s
                }
                """
            elif len(self.grouping_list) == 2:
                produce_template = """for (auto it=%(hashname)s.begin(); \
                it!=%(hashname)s.end(); it++) {
                %(output_tuple_type)s %(output_tuple_name)s(\
                {it->first.first, it->first.second, it->second});
                %(inner_code)s
                }
                """
        else:
            produce_template = """{
            %(output_tuple_type)s %(output_tuple_name)s({ %(hashname)s });
            %(inner_code)s
            }
            """

        output_tuple = CStagedTupleRef(gensym(), my_sch)
        output_tuple_name = output_tuple.name
        output_tuple_type = output_tuple.getTupleTypename()
        state.addDeclarations([output_tuple.generateDefinition()])

        inner_code = self.parent().consume(output_tuple, self, state)
        code = produce_template % locals()
        state.setPipelineProperty("type", "in_memory")
        state.addPipeline(code)

    def consume(self, inputTuple, fromOp, state):
        if self.useMap:
            if len(self.grouping_list) == 1:
                materialize_template = """%(op)s_insert(%(hashname)s, \
                %(tuple_name)s, %(key1pos)s, %(valpos)s);
                """
            elif len(self.grouping_list) == 2:
                materialize_template = """%(op)s_insert(%(hashname)s, \
                %(tuple_name)s, %(key1pos)s, %(key2pos)s, %(valpos)s);
                """
        else:
            materialize_template = """%(op)s_insert(%(hashname)s, \
            %(tuple_name)s, %(valpos)s);
            """

        hashname = self.hashname
        tuple_name = inputTuple.name

        # make key from grouped attributes
        if self.useMap:
            inp_sch = self.input.scheme()

            key1pos = self.grouping_list[0].get_position(inp_sch)

            if len(self.grouping_list) == 2:
                key2pos = self.grouping_list[1].get_position(
                    inp_sch)

        if isinstance(self.aggregate_list[0], expression.ZeroaryOperator):
            # no value needed for Zero-input aggregate,
            # but just provide the first column
            valpos = 0
        elif isinstance(self.aggregate_list[0], expression.UnaryOperator):
            # get value positions from aggregated attributes
            valpos = self.aggregate_list[0].input.get_position(self.scheme())
        else:
            assert False, "only support Unary or Zeroary aggregates"

        op = self.aggregate_list[0].__class__.__name__

        code = materialize_template % locals()
        return code


class CHashJoin(algebra.Join, CCOperator):
    _i = 0

    @classmethod
    def __genHashName__(cls):
        name = "hash_%03d" % cls._i
        cls._i += 1
        return name

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
        hashsym = state.lookupExpr((self.right, self.right_keypos))

        if not hashsym:
            # if right child never bound then store hashtable symbol and
            # call right child produce
            self._hashname = self.__genHashName__()
            _LOG.debug("generate hashname %s for %s", self._hashname, self)
            state.saveExpr((self.right, self.right_keypos), self._hashname)
            self.right.produce(state)
        else:
            # if found a common subexpression on right child then
            # use the same hashtable
            self._hashname = hashsym
            _LOG.debug("reuse hash %s for %s", self._hashname, self)

        self.left.childtag = "left"
        self.left.produce(state)

    def consume(self, t, src, state):
        if src.childtag == "right":
            declr_template = """std::unordered_map\
            <int64_t, std::vector<%(in_tuple_type)s>* > %(hashname)s;
            """

            right_template = """insert(%(hashname)s, %(keyname)s, %(keypos)s);
            """

            hashname = self._hashname
            keyname = t.name

            keypos = self.right_keypos

            in_tuple_type = t.getTupleTypename()

            # declaration of hash map
            hashdeclr = declr_template % locals()
            state.addDeclarations([hashdeclr])

            # materialization point
            code = right_template % locals()

            return code

        if src.childtag == "left":
            left_template = """
          for (auto %(right_tuple_name)s : \
          lookup(%(hashname)s, %(keyname)s.get(%(keypos)s))) {
            auto %(out_tuple_name)s = \
            combine<%(out_tuple_type)s> (%(keyname)s, %(right_tuple_name)s);
         %(inner_plan_compiled)s
      }
      """
            hashname = self._hashname
            keyname = t.name
            keytype = t.getTupleTypename()

            keypos = self.left_keypos

            right_tuple_name = gensym()

            outTuple = CStagedTupleRef(gensym(), self.scheme())
            out_tuple_type_def = outTuple.generateDefinition()
            out_tuple_type = outTuple.getTupleTypename()
            out_tuple_name = outTuple.name

            state.addDeclarations([out_tuple_type_def])

            inner_plan_compiled = self.parent().consume(outTuple, self, state)

            code = left_template % locals()
            return code

        assert False, "src not equal to left or right"


def indentby(code, level):
    indent = " " * ((level + 1) * 6)
    return "\n".join([indent + line for line in code.split("\n")])


# iteration  over table + insertion into hash table with filter

class CUnionAll(clangcommon.CUnionAll, CCOperator):
    pass


class CApply(clangcommon.CApply, CCOperator):
    pass


class CProject(clangcommon.CProject, CCOperator):
    pass


class CSelect(clangcommon.CSelect, CCOperator):
    pass


class CFileScan(clangcommon.CFileScan, CCOperator):
    def __get_ascii_scan_template__(self):
        return ascii_scan_template

    def __get_binary_scan_template__(self):
        return binary_scan_template


class CStore(clangcommon.BaseCStore, CCOperator):

    def __file_code__(self, t, state):
        code = ""
        state.addPreCode('std::ofstream logfile;\n')
        resultfile = str(self.relation_key).split(":")[2]
        opentuple = 'logfile.open("%s");\n' % resultfile
        schemafile = self.write_schema(self.scheme())
        state.addPreCode(schemafile)
        state.addPreCode(opentuple)
        code += "int logi = 0;\n"
        code += "for (logi = 0; logi < %s.numFields() - 1; logi++) {\n" \
                % (t.name)
        code += self.language().log_file_unquoted("%s.get(logi)" % t.name)
        code += "}\n "
        code += "logfile << %s.get(logi);\n" % (t.name)
        code += "logfile << '\\n';"
        state.addPostCode('logfile.close();')

        return code

    def write_schema(self, scheme):
        schemafile = 'schema/' + str(self.relation_key).split(":")[2]
        code = 'logfile.open("%s");\n' % schemafile
        names = [x.encode('UTF8') for x in scheme.get_names()]
        code += self.language().log_file("%s" % names)
        code += self.language().log_file("%s" % scheme.get_types())
        code += 'logfile.close();'
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
        # TODO: obviously breaks semantics
        rules.OneToOne(algebra.Union, CUnionAll),
        clangcommon.StoreToBaseCStore(emit_print, CStore),

        clangcommon.BreakHashJoinConjunction(CSelect, CHashJoin)
    ]


class CCAlgebra(Algebra):
    def __init__(self, emit_print=clangcommon.EMIT_CONSOLE):
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
            clangcommon.clang_push_select,
            [rules.ProjectToDistinctColumnSelect(),
             rules.JoinToProjectingJoin()],
            rules.push_apply,
            clangify(self.emit_print)
        ]

        if kwargs.get('SwapJoinSides'):
            rule_grps_sequence.insert(0, [rules.SwapJoinSides()])

        return list(itertools.chain(*rule_grps_sequence))
