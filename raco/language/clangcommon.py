import abc
from raco import algebra
from raco import expression
from raco import catalog
from raco.algebra import gensym
from raco.expression import UnnamedAttributeRef
from raco.language import Language
from raco.pipelines import Pipelined

import logging
_LOG = logging.getLogger(__name__)

import re


class CodeTemplate:
    def __init__(self, s):
        self.string = s

    @classmethod
    def __format_template__(cls, s):
        """
        Format code template string
        """
        return re.sub(r'[^\S\r\n]+', ' ', s)

    def __mod__(self, other):
        return (self.__format_template__(self.string)) % other


def ct(s):
    return CodeTemplate(s)


class CBaseLanguage(Language):
    @classmethod
    def body(cls, compileResult):
        queryexec = compileResult.getExecutionCode()
        initialized = compileResult.getInitCode()
        declarations = compileResult.getDeclCode()
        resultsym = "__result__"
        return cls.base_template() % locals()

    @staticmethod
    @abc.abstractmethod
    def base_template():
        pass

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
    def negation(cls, input):
        innerexpr, decls, inits = input
        return "(!%s)" % (innerexpr,), decls, inits

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
        _LOG.debug("conjunc: %s", conjunc)
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


# TODO:
# The following is actually a staged materialized tuple ref.
# we should also add a staged reference tuple ref that
# just has relationsymbol and row
class StagedTupleRef:
    nextid = 0

    @classmethod
    def genname(cls):
        # use StagedTupleRef so everyone shares one mutable copy of nextid
        x = StagedTupleRef.nextid
        StagedTupleRef.nextid += 1
        return "t_%03d" % x

    def __init__(self, relsym, scheme):
        self.name = self.genname()
        self.relsym = relsym
        self.scheme = scheme
        self.__typename = None

    def getTupleTypename(self):
        if self.__typename is None:
            fields = ""
            relsym = self.relsym
            for i in range(0, len(self.scheme)):
                fieldnum = i
                fields += "_%(fieldnum)s" % locals()

            self.__typename = "MaterializedTupleRef_%(relsym)s%(fields)s" \
                              % locals()

        return self.__typename

    def generateDefinition(self):
        fielddeftemplate = """int64_t _fields[%(numfields)s];
    """
        template = """
          // can be just the necessary schema
  class %(tupletypename)s {

    public:
    %(fielddefs)s

    int64_t get(int field) const {
      return _fields[field];
    }

    void set(int field, int64_t val) {
      _fields[field] = val;
    }

    int numFields() const {
      return %(numfields)s;
    }

    %(tupletypename)s () {
      // no-op
    }

    %(tupletypename)s (std::vector<int64_t> vals) {
      for (int i=0; i<vals.size(); i++) _fields[i] = vals[i];
    }

    std::ostream& dump(std::ostream& o) const {
      o << "Materialized(";
      for (int i=0; i<numFields(); i++) {
        o << _fields[i] << ",";
      }
      o << ")";
      return o;
    }

    %(additional_code)s
  } %(after_def_code)s;
  std::ostream& operator<< (std::ostream& o, const %(tupletypename)s& t) {
    return t.dump(o);
  }

  """
        getcases = ""
        setcases = ""
        copies = ""
        numfields = len(self.scheme)
        fielddefs = fielddeftemplate % locals()

        additional_code = self.__additionalDefinitionCode__()
        after_def_code = self.__afterDefinitionCode__()

        tupletypename = self.getTupleTypename()
        relsym = self.relsym

        code = template % locals()
        return code

    def __additionalDefinitionCode__(self):
        return ""

    def __afterDefinitionCode__(self):
        return ""


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


class CSelect(Pipelined, algebra.Select):
    def produce(self, state):
        self.input.produce(state)

    def consume(self, t, src, state):
        basic_select_template = """if (%(conditioncode)s) {
      %(inner_code_compiled)s
    }
    """

        condition_as_unnamed = expression.ensure_unnamed(self.condition, self)

        # tag the attributes with references
        # TODO: use an immutable approach instead
        # (ie an expression Visitor for compiling)
        [_ for _ in condition_as_unnamed.postorder(getTaggingFunc(t))]

        # compile the predicate into code
        conditioncode, cond_decls, cond_inits = \
            self.language().compile_expression(condition_as_unnamed)
        state.addInitializers(cond_inits)
        state.addDeclarations(cond_decls)

        inner_code_compiled = self.parent().consume(t, self, state)

        code = basic_select_template % locals()
        return code


class CUnionAll(Pipelined, algebra.Union):
    def produce(self, state):
        self.unifiedTupleType = self.new_tuple_ref(gensym(), self.scheme())
        state.addDeclarations([self.unifiedTupleType.generateDefinition()])

        self.right.produce(state)
        self.left.produce(state)

    def consume(self, t, src, state):
        union_template = """
        auto %(unified_tuple_name)s = \
        transpose<%(unified_tuple_typename)s>(%(src_tuple_name)s);
                        %(inner_plan_compiled)s"""

        unified_tuple_typename = self.unifiedTupleType.getTupleTypename()
        unified_tuple_name = self.unifiedTupleType.name
        src_tuple_name = t.name

        inner_plan_compiled = \
            self.parent().consume(self.unifiedTupleType, self, state)
        return union_template % locals()


class CApply(Pipelined, algebra.Apply):
    def produce(self, state):
        # declare a single new type for project
        # TODO: instead do mark used-columns?

        # always does an assignment to new tuple
        self.newtuple = self.new_tuple_ref(gensym(), self.scheme())
        state.addDeclarations([self.newtuple.generateDefinition()])

        self.input.produce(state)

    def consume(self, t, src, state):
        code = ""

        assignment_template = """
        %(dst_name)s.set(%(dst_fieldnum)s, %(src_expr_compiled)s);
        """

        dst_name = self.newtuple.name
        dst_type_name = self.newtuple.getTupleTypename()

        # declaration of tuple instance
        code += """%(dst_type_name)s %(dst_name)s;""" % locals()

        for dst_fieldnum, src_label_expr in enumerate(self.emitters):
            src_label, src_expr = src_label_expr

            # make sure to resolve attribute positions using input schema
            src_expr_unnamed = expression.ensure_unnamed(src_expr, self.input)

            # tag the attributes with references
            # TODO: use an immutable approach instead
            # (ie an expression Visitor for compiling)
            [_ for _ in src_expr_unnamed.postorder(getTaggingFunc(t))]

            src_expr_compiled, expr_decls, expr_inits = \
                self.language().compile_expression(src_expr_unnamed)
            state.addInitializers(expr_inits)
            state.addDeclarations(expr_decls)

            code += assignment_template % locals()

        innercode = self.parent().consume(self.newtuple, self, state)
        code += innercode

        return code


class CProject(Pipelined, algebra.Project):
    def produce(self, state):
        # declare a single new type for project
        # TODO: instead do mark used-columns?

        # always does an assignment to new tuple
        self.newtuple = self.new_tuple_ref(gensym(), self.scheme())
        state.addDeclarations([self.newtuple.generateDefinition()])

        self.input.produce(state)

    def consume(self, t, src, state):
        code = ""

        assignment_template = """
        %(dst_name)s.set(%(dst_fieldnum)s, %(src_name)s.get(%(src_fieldnum)s));
        """

        dst_name = self.newtuple.name
        dst_type_name = self.newtuple.getTupleTypename()
        src_name = t.name

        # declaration of tuple instance
        code += """%(dst_type_name)s %(dst_name)s;
        """ % locals()

        for dst_fieldnum, src_expr in enumerate(self.columnlist):
            if isinstance(src_expr, UnnamedAttributeRef):
                src_fieldnum = src_expr.position
            else:
                assert False, "Unsupported Project expression"
            code += assignment_template % locals()

        innercode = self.parent().consume(self.newtuple, self, state)
        code += innercode

        return code


from raco.algebra import ZeroaryOperator


class CFileScan(Pipelined, algebra.Scan):

    @abc.abstractmethod
    def __get_ascii_scan_template__(self):
        return

    @abc.abstractmethod
    def __get_binary_scan_template__(self):
        return

    def __get_relation_decl_template__(self, name):
        """
        Implement if the CFileScan implementation requires
        the relation instance to be a global declaration.
        If not then just put the local declaration within
        the *_scan_template.
        """
        return None

    def produce(self, state):

        # Common subexpression elimination
        # don't scan the same file twice
        resultsym = state.lookupExpr(self)
        _LOG.debug("lookup %s(h=%s) => %s", self, self.__hash__(), resultsym)
        if not resultsym:
            # TODO for now this will break
            # whatever relies on self.bound like reusescans
            # Scan is the only place where a relation is declared
            resultsym = gensym()

            name = str(self.relation_key).split(':')[2]
            fscode = self.__compileme__(resultsym, name)
            state.saveExpr(self, resultsym)

            stagedTuple = self.new_tuple_ref(resultsym, self.scheme())
            state.saveTupleDef(resultsym, stagedTuple)

            tuple_type_def = stagedTuple.generateDefinition()
            tuple_type = stagedTuple.getTupleTypename()
            state.addDeclarations([tuple_type_def])

            rel_decl_template = self.__get_relation_decl_template__(name)
            if rel_decl_template:
                state.addDeclarations([rel_decl_template % locals()])

            # now that we have the type, format this in;
            state.setPipelineProperty('type', 'scan')
            state.setPipelineProperty('source', self.__class__)
            state.addPipeline(fscode % {"result_type": tuple_type})

        # no return value used because parent is a new pipeline
        self.parent().consume(resultsym, self, state)

    def consume(self, t, src, state):
        assert False, "as a source, no need for consume"

    def __compileme__(self, resultsym, name):
        # TODO use the identifiers (don't split str and extract)
        # name = self.relation_key

        _LOG.debug('compiling file scan for relation_key %s'
                   % self.relation_key)

        # tup = (resultsym, self.originalterm.originalorder, self.originalterm)
        # self.trace("// Original query position of %s: term %s (%s)" % tup)

        if isinstance(self.relation_key, catalog.ASCIIFile):
            code = self.__get_ascii_scan_template__() % locals()
        else:
            code = self.__get_binary_scan_template__() % locals()
        return code

    def __str__(self):
        return "%s(%s)" % (self.opname(), self.relation_key)

    def __eq__(self, other):
        """
        For what we are using FileScan for, the only use
        of __eq__ is in hashtable lookups for CSE optimization.
        We omit self.schema because the relation_key determines
        the level of equality needed.

        This could break other things, so better may be to
        make a normalized copy of an expression. This could
        include simplification but in the case of Scans make
        the scheme more generic.

        @see MemoryScan.__eq__
        """
        return ZeroaryOperator.__eq__(self, other) and \
            self.relation_key == other.relation_key


# Rules
from raco import rules


class BreakHashJoinConjunction(rules.Rule):
    """A rewrite rule for turning HashJoin(a=c and b=d)
    into select(b=d)[HashJoin(a=c)]"""

    def __init__(self, select_clazz, join_clazz):
        self.select_clazz = select_clazz
        self.join_clazz = join_clazz

    def fire(self, expr):
        if isinstance(expr, self.join_clazz) \
                and isinstance(expr.condition.left, expression.EQ) \
                and isinstance(expr.condition.right, expression.EQ):
            return self.select_clazz(expr.condition.right,
                                     self.join_clazz(expr.condition.left,
                                                     expr.left,
                                                     expr.right))

        return expr

    def __str__(self):
        return "%s(a=c and b=d) => %s(b=d)[%s(a=c)]" \
               % (self.join_clazz.__name__,
                  self.select_clazz.__name__,
                  self.join_clazz.__name__)


clang_push_select = [
    rules.SplitSelects(),
    rules.PushSelects(),
    # We don't want to merge selects because it doesn't really
    # help and it (maybe) creates HashJoin(conjunction)
    # rules.MergeSelects()
]

EMIT_CONSOLE = 'console'
EMIT_FILE = 'file'


class BaseCStore(Pipelined, algebra.Store):
    def __init__(self, emit_print, relation_key, plan):
        super(BaseCStore, self).__init__(relation_key, plan)
        self.emit_print = emit_print

    def produce(self, state):
        self.input.produce(state)

    def consume(self, t, src, state):
        code = ""
        resdecl = "std::vector<%s> result;\n" % (t.getTupleTypename())
        state.addDeclarations([resdecl])
        code += "result.push_back(%s);\n" % (t.name)

        if self.emit_print == EMIT_CONSOLE:
            code += self.language().log_unquoted("%s" % t.name, 2)
        elif self.emit_print == EMIT_FILE:
            code += self.__file_code__(t, state)

        return code

    @abc.abstractmethod
    def __file_code__(self, t, state):
        pass

    def __repr__(self):
        return "{op}({ep!r}, {rk!r}, {pl!r})".format(op=self.opname(),
                                                     ep=self.emit_print,
                                                     rk=self.relation_key,
                                                     pl=self.input)


class StoreToBaseCStore(rules.Rule):
    """A rule to store tuples into emit_print"""
    def __init__(self, emit_print, subclass):
        self.emit_print = emit_print
        assert issubclass(subclass, BaseCStore), \
            "%s is not a subclass of %s" % (subclass, BaseCStore)
        self.subclass = subclass

    def fire(self, expr):
        if isinstance(expr, algebra.Store):
            return self.subclass(self.emit_print,
                                 expr.relation_key,
                                 expr.input)
        return expr

    def __str__(self):
        return "Store => %s" % self.subclass.__name__
