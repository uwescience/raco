import abc
import itertools
import re
import os.path

from raco import algebra
from raco import expression
from raco import catalog
from raco.algebra import gensym
from raco.expression import UnnamedAttributeRef
from raco.language import Language
from raco.pipelines import Pipelined
from raco.utility import emitlist
from raco import types

import logging
_LOG = logging.getLogger(__name__)


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


def readtemplate(grouppath, fname):
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 grouppath)

    return file(os.path.join(template_path, fname+'.template')).read()


class CBaseLanguage(Language):
    @classmethod
    def body(cls, compileResult):
        queryexec = compileResult.getExecutionCode()
        initialized = compileResult.getInitCode()
        declarations = compileResult.getDeclCode()
        resultsym = "__result__"
        return cls.base_template() % locals()

    @classmethod
    @abc.abstractmethod
    def base_template(cls):
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

    @staticmethod
    def _extract_code_decl_init(args):
        codes = [c for c, _, _ in args]
        decls_combined = reduce(lambda sofar, x: sofar + x, [d for _, d, _ in args])
        inits_combined = reduce(lambda sofar, x: sofar + x, [i for _, _, i in args])
        return codes, decls_combined, inits_combined

    @classmethod
    def expression_combine(cls, args, operator="&&"):
        opstr = " %s " % operator
        codes, decls, inits = cls._extract_code_decl_init(args)
        conjunc = opstr.join(["(%s)" % c for c in codes])
        _LOG.debug("conjunc: %s", conjunc)
        return "( %s )" % conjunc, decls, inits

    @classmethod
    def function_call(cls, name, *args):
        codes, decls, inits = cls._extract_code_decl_init(list(args))
        argscode = ",".join(["{0}".format(d) for d in codes])
        code = "{name}({argscode})".format(name=name.lower(), argscode=argscode)
        return code, decls, inits

    @classmethod
    def typename(cls, raco_type):
        n = {
            types.LONG_TYPE: 'int64_t',
            types.BOOLEAN_TYPE: 'bool',
            types.DOUBLE_TYPE: 'double'
        }.get(raco_type)

        assert n is not None, "Clang does not yet support type {type}".format(type=n)
        return n

    @classmethod
    def cast(cls, castto, inputexpr):
        inputcode, decls, inits = inputexpr
        typen = cls.typename(castto)
        code = "(({typename}){expr})".format(typename=typen, expr=inputcode)
        return code, decls, inits

    @classmethod
    def compile_attribute(cls, expr, **kwargs):
        if isinstance(expr, expression.NamedAttributeRef):
            raise TypeError(
                "Error compiling attribute reference %s. \
                C compiler only support unnamed perspective. \
                Use helper function unnamed." % expr)
        if isinstance(expr, expression.UnnamedAttributeRef):
            tupleref = kwargs.get('tupleref')
            assert tupleref is not None, "Cannot compile {0} without a tupleref".format(expr)

            position = expr.position
            assert position >= 0
            return tupleref.get_code(position), [], []
        if isinstance(expr, expression.NamedStateAttributeRef):
            state_scheme = kwargs.get('state_scheme')
            assert state_scheme is not None, "Cannot compile {0} without a state_scheme".format(expr)

            position = expr.get_position(None, state_scheme)
            code = StagedTupleRef.get_code_with_name(position, "state")
            return code, [], []

        assert False, "{expr} is unsupported attribute".format(expr=expr)

    @classmethod
    def ifelse(cls, when_compiled, else_compiled):
        if_template = """
        if ({cond}) {{
        {then}
        }}
        """

        else_template = """
        else {{
        {then}
        }}
        """

        return cls._conditional_(when_compiled, else_compiled, if_template, else_template, "else")

    @classmethod
    def limits(cls, side, typ):
        return "std::numeric_limits<{type}>::{side}()".format(type=cls.typename(typ),
                                                              side=side)

    @classmethod
    def conditional(cls, when_compiled, else_compiled):
        if_template = """{cond} ? {then}"""

        else_template = """: {then}"""

        return cls._conditional_(when_compiled, else_compiled, if_template, else_template, ":")

    @classmethod
    def _conditional_(cls, when_compiled, else_compiled, if_template, else_template, else_joiner):
        def by_pairs(l):
            assert len(l) % 2 == 0, "must be even length"
            for i in range(0, len(l), 2):
                yield l[i], l[i+1]

        flatten_when_then = list(itertools.chain.from_iterable(when_compiled))
        when_exec, when_decls, when_inits = zip(*flatten_when_then)

        code = else_joiner.join([if_template.format(cond=cond, then=then) for (cond, then) in by_pairs(when_exec)])

        if else_compiled is not None:
            code += else_template.format(then=else_compiled[0])

        return code, \
               list(itertools.chain.from_iterable(when_decls))+else_compiled[1], \
               list(itertools.chain.from_iterable(when_inits))+else_compiled[2]


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

    @staticmethod
    def get_code_with_name(position, name):
        return "{name}.get<{position}>()".format(position=position, name=name)

    def get_code(self, position):
        return StagedTupleRef.get_code_with_name(position, self.name)

    def set_func_code(self, position):
        return "{name}.set<{position}>".format(position=position, name=self.name)

    def generateDefinition(self):
        template = readtemplate('c_templates', 'materialized_tuple_ref')
        numfields = len(self.scheme)

        fieldtypes = ','.join([CBaseLanguage.typename(t) for t in self.scheme.get_types()])
        string_append_statements = emitlist(['o << std::get<{i}>(_fields) << ",";'.format(i=i) for i in range(numfields)])

        stream_reads = "ss" + emitlist([' >> std::get<{i}>(_t._fields)'.format(i=i) for i in range(numfields)]) + ";"

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


class CSelect(Pipelined, algebra.Select):
    def produce(self, state):
        self.input.produce(state)

    def consume(self, t, src, state):
        basic_select_template = """if (%(conditioncode)s) {
      %(inner_code_compiled)s
    }
    """

        condition_as_unnamed = expression.ensure_unnamed(self.condition, self)

        # compile the predicate into code
        conditioncode, cond_decls, cond_inits = \
            self.language().compile_expression(condition_as_unnamed, tupleref=t)
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
        %(unified_tuple_typename)s %(unified_tuple_name)s = \
        %(unified_tuple_typename)s::create(%(src_tuple_name)s);
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
        %(dst_set_func)s(%(src_expr_compiled)s);
        """

        dst_name = self.newtuple.name
        dst_type_name = self.newtuple.getTupleTypename()

        # declaration of tuple instance
        code += """%(dst_type_name)s %(dst_name)s;""" % locals()

        for dst_fieldnum, src_label_expr in enumerate(self.emitters):
            dst_set_func = self.newtuple.set_func_code(dst_fieldnum)
            src_label, src_expr = src_label_expr

            # make sure to resolve attribute positions using input schema
            src_expr_unnamed = expression.ensure_unnamed(src_expr, self.input)

            src_expr_compiled, expr_decls, expr_inits = \
                self.language().compile_expression(src_expr_unnamed, tupleref=t)
            state.addInitializers(expr_inits)
            state.addDeclarations(expr_decls)

            print locals()
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
        %(dst_set_func)s(%(src_val)s);
        """

        dst_name = self.newtuple.name
        dst_type_name = self.newtuple.getTupleTypename()

        # declaration of tuple instance
        code += """%(dst_type_name)s %(dst_name)s;
        """ % locals()

        for dst_fieldnum, src_expr in enumerate(self.columnlist):
            if isinstance(src_expr, UnnamedAttributeRef):
                src_val = t.get_code(src_expr.position)
                dst_set_func = t.set_func_code(dst_fieldnum)
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


class BaseCGroupby(Pipelined, algebra.GroupBy):
    def __get_initial_value__(self, index, cached_inp_sch=None):
        if cached_inp_sch is None:
            cached_inp_sch = self.input.scheme()

        op = self.aggregate_list[index].__class__.__name__

        # min, max need special values; default to 0 as initial value
        initial_value = {'MAX': self.language().limits('min', self.aggregate_list[index].typeof(cached_inp_sch, None)),
                         'MIN': self.language().limits('max', self.aggregate_list[index].typeof(cached_inp_sch, None))}.get(op, 0)
        return initial_value
