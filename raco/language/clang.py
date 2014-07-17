# TODO: To be refactored into shared memory lang,
# where you plugin in the sequential shared memory language specific codegen

from raco import algebra
from raco import expression
from raco.language import Language, clangcommon
from raco import rules
from raco.pipelines import Pipelined
from raco.language.clangcommon import StagedTupleRef, ct

from raco.algebra import gensym

import logging

LOG = logging.getLogger(__name__)

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


class CC(Language):
    @classmethod
    def new_relation_assignment(cls, rvar, val):
        return """
    %s
    %s
    """ % (cls.relation_decl(rvar), cls.assignment(rvar, val))

    @classmethod
    def relation_decl(cls, rvar):
        return "struct relationInfo *%s;" % rvar

    @classmethod
    def assignment(cls, x, y):
        return "%s = %s;" % (x, y)

    @staticmethod
    def body(compileResult):
        queryexec = compileResult.getExecutionCode()
        initialized = compileResult.getInitCode()
        declarations = compileResult.getDeclCode()
        resultsym = "__result__"
        return base_template % locals()

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
        return """logfile << %s << "\\n";\n """ % code

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

    @classmethod
    def negation(cls, input):
        innerexpr, inits = input
        return "(!%s)" % (innerexpr,), [], inits

    @classmethod
    def negative(cls, input):
        innerexpr, decls, inits = input
        return "(-%s)" % (innerexpr,), decls, inits

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
            raise TypeError(
                "Error compiling attribute reference %s. \
                C compiler only support unnamed perspective. \
                Use helper function unnamed." % expr)
        if isinstance(expr, expression.UnnamedAttributeRef):
            symbol = expr.tupleref.name
            position = expr.position
            assert position >= 0
            return '%s.get(%s)' % (symbol, position), [], []


class CCOperator(Pipelined):
    language = CC

    def new_tuple_ref(self, sym, scheme):
        return CStagedTupleRef(sym, scheme)


from raco.algebra import UnaryOperator


class MemoryScan(algebra.UnaryOperator, CCOperator):
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

        inner_plan_compiled = self.parent.consume(stagedTuple, self, state)

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
        assert len(self.grouping_list) <= 1, \
            "%s does not currently support groupings of \
            more than 1 attribute" % self.__class__.__name__
        assert len(self.aggregate_list) == 1, \
            """%s currently only supports aggregates of 1 attribute
            (aggregate_list=%s)""" \
            % (self.__class__.__name__, self.aggregate_list)
        for agg_term in self.aggregate_list:
            assert isinstance(agg_term, expression.AggregateExpression), \
                """%s only supports simple aggregate expressions.
                A rule should create Apply[GroupBy]""" \
                % self.__class__.__name__

        self.useMap = len(self.grouping_list) > 0

        if self.useMap:
            declr_template = """std::unordered_map<int64_t, int64_t> \
            %(hashname)s;
      """
        else:
            declr_template = """int64_t %(hashname)s;
            """

        self.hashname = self.__genHashName__()
        hashname = self.hashname

        hash_declr = declr_template % locals()
        state.addDeclarations([hash_declr])

        LOG.debug("aggregates: %s", self.aggregate_list)
        LOG.debug("columns: %s", self.column_list())
        LOG.debug("groupings: %s", self.grouping_list)
        LOG.debug("groupby scheme: %s", self.scheme())
        LOG.debug("groupby scheme[0] type: %s", type(self.scheme()[0]))

        self.input.produce(state)

        # now that everything is aggregated, produce the tuples
        assert (not self.useMap) \
            or isinstance(self.column_list()[0],
                          expression.UnnamedAttributeRef), \
            "assumes first column is the key and second is aggregate result"

        if self.useMap:
            produce_template = """for (auto it=%(hashname)s.begin(); \
            it!=%(hashname)s.end(); it++) {
            %(output_tuple_type)s %(output_tuple_name)s(\
            {it->first, it->second});
            %(inner_code)s
            }
            """
        else:
            produce_template = """{
            %(output_tuple_type)s %(output_tuple_name)s({ %(hashname)s });
            %(inner_code)s
            }
            """

        output_tuple = CStagedTupleRef(gensym(), self.scheme())
        output_tuple_name = output_tuple.name
        output_tuple_type = output_tuple.getTupleTypename()
        state.addDeclarations([output_tuple.generateDefinition()])

        inner_code = self.parent.consume(output_tuple, self, state)
        code = produce_template % locals()
        state.setPipelineProperty("type", "in_memory")
        state.addPipeline(code)

    def consume(self, inputTuple, fromOp, state):
        if self.useMap:
            materialize_template = """%(op)s_insert(%(hashname)s, \
            %(tuple_name)s, %(keypos)s, %(valpos)s);
      """
        else:
            materialize_template = """%(op)s_insert(%(hashname)s, \
            %(tuple_name)s, %(valpos)s);
            """

        hashname = self.hashname
        tuple_name = inputTuple.name

        # make key from grouped attributes
        if self.useMap:
            keypos = self.grouping_list[0].get_position(self.scheme())

        # get value positions from aggregated attributes
        valpos = self.aggregate_list[0].input.get_position(self.scheme())

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

        # find the attribute that corresponds to the right child
        self.rightCondIsRightAttr = \
            self.condition.right.position >= len(self.left.scheme())
        self.leftCondIsRightAttr = \
            self.condition.left.position >= len(self.left.scheme())
        assert self.rightCondIsRightAttr ^ self.leftCondIsRightAttr

        # find the attribute that corresponds to the right child
        if self.rightCondIsRightAttr:
            self.right_keypos = \
                self.condition.right.position - len(self.left.scheme())
        else:
            self.right_keypos = \
                self.condition.left.position - len(self.left.scheme())

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
            LOG.debug("generate hashname %s for %s", self._hashname, self)
            state.saveExpr((self.right, self.right_keypos), self._hashname)
            self.right.produce(state)
        else:
            # if found a common subexpression on right child then
            # use the same hashtable
            self._hashname = hashsym
            LOG.debug("reuse hash %s for %s", self._hashname, self)

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

            inner_plan_compiled = self.parent.consume(outTuple, self, state)

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


class CStore(algebra.Store, CCOperator):
    def __init__(self, emit_print, relation_key, plan):
        super(CStore, self).__init__(relation_key, plan)
        self.emit_print = emit_print

    def produce(self, state):
        self.input.produce(state)

    def consume(self, t, src, state):
        code = ""
        resdecl = "std::vector<%s> result;\n" % (t.getTupleTypename())
        state.addDeclarations([resdecl])
        code += "result.push_back(%s);\n" % (t.name)

        if self.emit_print in ['console', 'both']:
            code += self.language.log_unquoted("%s" % t.name, 2)
        if self.emit_print in ['file', 'both']:
            state.addPreCode('std::ofstream logfile;\n')
            filename = 'datasets/' + str(self.relation_key) + '.txt'
            openfile = 'logfile.open("%s", std::ios::app);\n' % filename
            state.addPreCode(openfile)
            code += self.language.log_file("%s" % t.name, 2)
            state.addPostCode('logfile.close();')
        return code


class MemoryScanOfFileScan(rules.Rule):
    """A rewrite rule for making a scan into
    materialization in memory then memory scan"""

    def fire(self, expr):
        if isinstance(expr, algebra.Scan) and not isinstance(expr, CFileScan):
            return MemoryScan(CFileScan(expr.relation_key, expr.scheme()))
        return expr

    def __str__(self):
        return "Scan => MemoryScan(FileScan)"


class StoreToCStore(rules.Rule):
    """A rule to store tuples into emit_print"""
    def __init__(self, emit_print):
        self.emit_print = emit_print

    def fire(self, expr):
        if isinstance(expr, algebra.Store):
            return CStore(self.emit_print, expr.relation_key, expr.input)
        return expr

    def __str__(self):
        return "Store => CStore"


class CCAlgebra(object):
    language = CC

    operators = [
        # TwoPassHashJoin,
        # FilteringNestedLoopJoin,
        # TwoPassSelect,
        # FileScan,
        MemoryScan,
        CSelect,
        CUnionAll,
        CApply,
        CProject,
        CGroupBy,
        CHashJoin,
        CStore
    ]

    def __init__(self, emit_print='console'):
        """ To store results into a file, onto console, both file and console,
        or stays quiet """
        self.emit_print = emit_print

    def opt_rules(self):
        return [
            # rules.OneToOne(algebra.Join,TwoPassHashJoin),
            # rules.removeProject(),
            rules.CrossProduct2Join(),
            rules.SimpleGroupBy(),
            #    FilteringNestedLoopJoinRule(),
            #    FilteringHashJoinChainRule(),
            #    LeftDeepFilteringJoinChainRule(),
            rules.OneToOne(algebra.Select, CSelect),
            #   rules.OneToOne(algebra.Select,TwoPassSelect),
            #  rules.OneToOne(algebra.Scan,MemoryScan),
            MemoryScanOfFileScan(),
            rules.OneToOne(algebra.Apply, CApply),
            rules.OneToOne(algebra.Join, CHashJoin),
            rules.OneToOne(algebra.GroupBy, CGroupBy),
            rules.OneToOne(algebra.Project, CProject),
            # TODO: obviously breaks semantics
            rules.OneToOne(algebra.Union, CUnionAll),
            StoreToCStore(self.emit_print)
            #  rules.FreeMemory()


        ]
