
import abc
from raco.utility import emitlist
from algebra import gensym, Operator
import re

import logging
LOG = logging.getLogger(__name__)


class ResolvingSymbol:
    _unique_tag = "_@@UNRESOLVED_SYMBOL_{name}@@_"

    def __init__(self, name):
        self._name = name
        self._placeholder = ResolvingSymbol._mksymbol(name)

    @classmethod
    def _mksymbol(cls, name):
        return cls._unique_tag.format(name=name)

    def getPlaceholder(self):
        return self._placeholder

    def getName(self):
        return self._name

    @classmethod
    def substitute(cls, code, symbols):
        # inefficient multi-string replacement
        # TODO: replace with multi-pattern sed script
        for name, val in symbols.items():
            assert val is not None, "Unresolved symbol: {}".format(name)
            pat = cls._unique_tag.format(name=name)
            code = re.sub(pat, val, code)
        return code


class CompileState(object):

    def __init__(self, lang, cse=True):
        self.language = lang

        self.declarations = []
        self.declarations_later = []
        self.pipelines = []
        self.scan_pipelines = []
        self.flush_pipelines = []
        self.initializers = []
        self.cleanups = []
        self.pipeline_count = 0

        # { expression => symbol for materialized result }
        self.materialized = {}

        # { symbol => tuple type definition }
        self.tupledefs = {}

        # symbol resolution
        self.resolving_symbols = {}

        self.common_subexpression_elim = cse

        self.current_pipeline_properties = {}
        self.current_pipeline_precode = []
        self.current_pipeline_postcode = []

        self.sequence_wait_statements = set()

        self.in_loop = False
        # this is a set for cases where we might generate recycle twice
        # as in operators above a union or other streaming binary operator
        self.loop_recycle_codes = set()
        self.loop_pipeline_codes = []

    def recordCodeWhenInLoop(self, code):
        if self.in_loop:
            self.loop_recycle_codes.add(code)

    def enterLoop(self):
        assert not self.in_loop, "Nested loops not supported"
        self.loop_recycle_codes = set()
        self.loop_pipeline_codes = []
        self.in_loop = True

    def exitLoop(self):
        assert self.in_loop, "Exiting loop but not in one"
        self.in_loop = False
        return self.loop_pipeline_codes, self.loop_recycle_codes

    def setPipelineProperty(self, key, value):
        LOG.debug("set %s in %s" % (key, self.current_pipeline_properties))
        self.current_pipeline_properties[key] = value

    def addToPipelinePropertyList(self, key, value):
        current = self.current_pipeline_properties.get(key, [])
        assert isinstance(current, list), "cannot add to non-list property"
        if len(current) == 0:
            self.current_pipeline_properties[key] = current

        current.append(value)

    def addToPipelinePropertySet(self, key, value):
        current = self.current_pipeline_properties.get(key, set())
        assert isinstance(current, set), "cannot add to non-set property"
        if len(current) == 0:
            self.current_pipeline_properties[key] = current

        current.add(value)

    def getPipelineProperty(self, key):
        LOG.debug("get %s from %s" % (key, self.current_pipeline_properties))
        return self.current_pipeline_properties[key]

    def checkPipelineProperty(self, key):
        """
        Like getPipelineProperty but returns None if no property is found
        """
        LOG.debug("get(to check) %s from %s"
                  % (key, self.current_pipeline_properties))
        return self.current_pipeline_properties.get(key)

    def createUnresolvedSymbol(self):
        name = gensym()
        rs = ResolvingSymbol(name)
        self.resolving_symbols[name] = None
        return rs

    def resolveSymbol(self, rs, value):
        self.resolving_symbols[rs.getName()] = value

    def resolveCounterSymbol(self, rs, increment=1):
        """For late-resolved symbols that are incremented integers whose
        value is unknown until a later compilation step"""

        # Only store strings in resolving_symbols to maintain symbol is
        # a string invariant. We just convert back and forth here to increment
        if self.resolving_symbols[rs.getName()] is None:
            self.resolving_symbols[rs.getName()] = '0'

        self.resolving_symbols[rs.getName()] = str(
            int(self.resolving_symbols[rs.getName()]) + increment)

    def addDeclarations(self, d):
        self.declarations += d

    def addDeclarationsUnresolved(self, d):
        """
        Ordered in the code after the regular declarations
        just so that any name dependences already have been declared
        ALTERNATIVE: split decls into forward decls and definitions
        """
        self.declarations_later += d

    def addInitializers(self, i):
        self.initializers += i

    def addCleanups(self, i):
        self.cleanups += i

    def addSeqWaitStatement(self, c):
        self.sequence_wait_statements.add(c)

    def getAndFlushSeqWaitStatements(self):
        r = self.sequence_wait_statements
        self.sequence_wait_statements = set()
        return r

    def _append_pipeline_code(self, code):
        if self.in_loop:
            self.loop_pipeline_codes.append(code)
        else:
            self.pipelines.append(code)

    def addPipeline(self, p):
        LOG.debug("output pipeline %s", self.current_pipeline_properties)

        all_p = emitlist(self.current_pipeline_precode) \
            + p \
            + emitlist(self.current_pipeline_postcode)

        pipeline_code = \
            self.language.pipeline_wrap(self.pipeline_count, all_p,
                                        self.current_pipeline_properties)

        # force scan pipelines to go first
        if self.current_pipeline_properties.get('type') == 'scan':
            assert not self.in_loop, "scan pipeline not supported in loop"
            self.scan_pipelines.append(pipeline_code)
        else:
            self._append_pipeline_code(pipeline_code)

        self.pipeline_count += 1
        self.current_pipeline_properties = {}
        self.current_pipeline_precode = []
        self.current_pipeline_postcode = []

    def addCode(self, c):
        """
        Just add code here
        """
        self._append_pipeline_code(c)

    def addPreCode(self, c):
        self.current_pipeline_precode.append(c)

    def addPostCode(self, c):
        self.current_pipeline_postcode.append(c)

    def addPipelineFlushCode(self, c):
        self.flush_pipelines.append(c)

    def getInitCode(self):

        # inits is a set.
        # If this ever becomes a bottleneck when declarations are strings,
        # as in cpp, then resort to at least symbol name deduping.
        # TODO: better would be to mark elements of self.initializers as
        # TODO: "do dedup" or "don't dedup"
        s = set()

        def f(x):
            if x in s:
                return False
            else:
                s.add(x)
                return True

        code = emitlist(filter(f, self.initializers))
        return ResolvingSymbol.substitute(code, self.resolving_symbols)

    def getCleanupCode(self):
        # cleanups is a set.
        # If this ever becomes a bottleneck when declarations are strings,
        # as in cpp, then resort to at least symbol name deduping.
        # TODO: better would be to mark elements of self.cleanups as
        # TODO: "do dedup" or "don't dedup"
        s = set()

        def f(x):
            if x in s:
                return False
            else:
                s.add(x)
                return True

        code = emitlist(filter(f, self.cleanups))
        return ResolvingSymbol.substitute(code, self.resolving_symbols)

    def getDeclCode(self):
        # declarations is a set
        # If this ever becomes a bottleneck when declarations are strings,
        # as in cpp, then resort to at least symbol name deduping.
        s = set()

        def f(x):
            if x in s:
                return False
            else:
                s.add(x)
                return True

        # keep in original order
        code = emitlist(filter(f, self.declarations))
        code += emitlist(filter(f, self.declarations_later))
        return ResolvingSymbol.substitute(code, self.resolving_symbols)

    def getExecutionCode(self):
        assert not self.in_loop, "Bad state, in loop at end of compilation"

        # list -> string
        scan_linearized = emitlist(self.scan_pipelines)

        # Make sure we emitted all the wait statements
        if self.sequence_wait_statements:
            waits_linearized = emitlist(
                list(self.getAndFlushSeqWaitStatements()))
        else:
            waits_linearized = ""

        mem_linearized = \
            emitlist(self.pipelines) + waits_linearized

        flush_linearized = emitlist(self.flush_pipelines)
        scan_linearize_wrap = self.language.group_wrap(gensym(),
                                                       scan_linearized,
                                                       {'type': 'scan'})
        mem_linearize_wrap = self.language.group_wrap(gensym(),
                                                      mem_linearized,
                                                      {'type': 'in_memory'})

        linearized = \
            scan_linearize_wrap + mem_linearize_wrap + flush_linearized

        # substitute all lazily resolved symbols
        resolved = ResolvingSymbol.substitute(linearized,
                                              self.resolving_symbols)

        return resolved

    def lookupExpr(self, expr):

        if self.common_subexpression_elim:
            res = self.materialized.get(expr)
            LOG.debug("lookup subexpression %s -> %s", expr, res)
            return res
        else:
            # if CSE is turned off
            # then always return None for expression matches
            return None

    def saveExpr(self, expr, sym):
        LOG.debug("saving subexpression %s -> %s", expr, sym)
        self.materialized[expr] = sym

    def lookupTupleDef(self, sym):
        return self.tupledefs.get(sym)

    def saveTupleDef(self, sym, tupledef):
        self.tupledefs[sym] = tupledef

    def getCurrentPipelineId(self):
        return self.pipeline_count


class Pipelined(object):

    """
    Trait to provide the compilePipeline method
    for calling into pipeline style compilation.

    This is a mixin class that supports cooperative
    multiple inheritance. To use it properly, put
    it _before_ the base class in the inheritance clause
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, *args):
        self._parent = None
        # Ensure this class follows cooperative multiple inheritance
        super(Pipelined, self).__init__(*args)

    def _markAllParents(self, test_mode=False, **kwargs):
        root = self

        def markChildParent(op):
            for c in op.children():
                c._parent = op
                if test_mode:
                    # Test mode turns on assign-once checks of all
                    # Pipeline subclasses instance variables
                    c._freeze()  # _parent is the last allowed attribute
            return []

        [_ for _ in root.postorder_traversal(markChildParent)]

    __isfrozen = False

    def __setattr__(self, key, value):
        """Overriden to allow objects to turn on assigned-once checks

        While we'd prefer subclasses of Pipelined (more specifically Operator)
        to actually be immutable algebraic datatypes, this is an intermediate
        solution. It was introduced to catch bugs that occur from
        reassigning instance variables that are introduced for compilation
        state (see issue https://github.com/uwescience/raco/issues/477).
        """
        if self.__isfrozen and key in self.assigned_attrs:
            if self.__getattribute__(key) == value:
                LOG.warning(
                    '''reassignment of self.{attr} but ignoring because
                    assigned same value {value}'''.format(
                        attr=key,
                        value=value))
                return
            else:
                raise TypeError(
                    """{obj} is a frozen object; self.{attr} = {oldval};
                    tried to assign {newval}""".format(
                        obj=self,
                        attr=key,
                        oldval=self.__getattribute__(key),
                        newval=value))
        # new set created here rather than __init__ because there
        # may be inconsistency in when Pipelined.__init__ is called relative
        # to assignments to instance variables
        if not hasattr(self, 'assigned_attrs'):
            object.__setattr__(self, 'assigned_attrs', set())
        self.assigned_attrs.add(key)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def parent(self):
        return self._parent

    def run_analyses(self, **kwargs):
        """Intended to be called on the root of the tree"""
        self._markAllParents(**kwargs)

    @classmethod
    @abc.abstractmethod
    def language(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def new_tuple_ref(cls, symbol, scheme):
        pass

    @abc.abstractmethod
    def postorder_traversal(self, func):
        pass

    @abc.abstractmethod
    def produce(self, state):
        """Denotation for producing a tuple"""
        return

    @abc.abstractmethod
    def consume(self, inputTuple, fromOp, state):
        """Denotation for consuming a tuple"""
        return

    def compilePipeline(self, compiler='push', **kwargs):
        # run analyses
        self.run_analyses(**kwargs)

        compilerstate = {'push': CompileState,
                         'iterator': IteratorCompileState
                         }[compiler]
        state = compilerstate(self.language())

        state.addCode(
            self.language().comment("Compiled subplan for %s" % self))

        self.produce(state)

        # state.addCode( self.language().log("Evaluating subplan %s" % self) )

        return state


class IteratorCompileState(CompileState):

    def __init__(self, lang, cse=True):
        super(IteratorCompileState, self).__init__(lang, cse)
        self.iterator_operators = []

    def addOperator(self, code):
        self.iterator_operators.append(code)

    def addPipeline(self, p=None):
        # base class addPipeline takes code from the client,
        # but the IteratorCompileState keeps track of the code itself in
        # self.iterator_operators
        if p is None:  # None indicates an iterator pipeline
            p = self.language.iterators_wrap(
                ''.join(
                    self.iterator_operators),
                self.current_pipeline_properties)

        super(IteratorCompileState, self).addPipeline(p)
        self.iterator_operators = []
