import logging

import raco.rules

from raco import algebra
from raco.scheme import Scheme
from raco.backends import Language, Algebra
from raco.expression import UnnamedAttributeRef, \
    NamedAttributeRef, \
    AttributeRef

LOGGER = logging.getLogger(__name__)

# for gensym
counter = 0


class SPARQLLanguage(Language):
    EQ = "="

    @classmethod
    def body(cls, expr):
        return expr.compile()

    @classmethod
    def compile_attribute(cls, expr):
        if isinstance(expr, NamedAttributeRef):
            var = expr
        if isinstance(expr, UnnamedAttributeRef):
            var = cls.currentop.scheme().getName(expr.position)
        return "?%s" % var

    @classmethod
    def compile_stringliteral(cls, value):
        return str(value).replace('"', '')


class SPARQLOperator(object):
    language = SPARQLLanguage


class SPARQLUnionAll(algebra.UnionAll, SPARQLOperator):

    def compile(self):
        return """
      {
      %s
      }
      UNION
      {
      %s
      }
      """ % (self.left.compile(), self.right.compile())


class SPARQLScan(algebra.Scan, SPARQLOperator):

    def renameattrs(self):
        """Make attribute names globally unique so they
        can be used as SPARQL variables"""
        global counter
        c = counter
        counter = counter + 1
        self._scheme = Scheme([("%s%s" % (n, c), typ)
                               for n, typ in self._scheme])

    def __init__(self, *args):
        # Python 3.0 cleans this crap up
        super(self.__class__, self).__init__(*args)
        if self._scheme:
            self.renameattrs()

    def copy(self, other):
        # Python 3.0 cleans this crap up
        super(self.__class__, self).copy(other)
        self.renameattrs()

    def compile(self):
        names = self.scheme().get_names()
        triplepattern = " ".join(["?%s" % n for n in names])
        return "%s ." % (triplepattern)


class SPARQLSelect(algebra.Select, SPARQLOperator):

    def compile(self):
        # This is pretty bad: passing information to compile_expression
        # by way of class instance variable
        # The right thing to do seems to be to arrange for every
        # AttributeRef to hold a pointer to the operator
        # whose result is being referenced.  This way, if
        # you have an AttributeRef in hand,
        # you know you can do something useful with it.
        self.language.currentop = self
        filterexpr = self.language.compile_expression(self.condition)
        return """%s
    FILTER (%s) .""" % (self.input.compile(), filterexpr)


class SPARQLStore(algebra.Store, SPARQLOperator):

    def compile(self):
        return """%s""" % (self.input.compile())


class SPARQLStoreTemp(SPARQLStore):
    pass


class SPARQLJoin(algebra.Join, SPARQLOperator):

    def compile(self):
        leftc = self.left.compile()
        rightc = self.right.compile()
        self.language.currentop = self
        joincond = self.language.compile_expression(self.condition)
        return """
        {
        %s
        }
        {
        %s
        }
        FILTER (%s)
        """ % (self.left.compile(), self.right.compile(), joincond)


class SPARQLApply(algebra.Apply, SPARQLOperator):

    """Represents a simple apply operator"""

    def compile(self):
        child_scheme = self.input.scheme()
        self.language.currentop = self.input

        def formatemitter(name, exp):
            e = self.language.compile_expression(exp)
            if isinstance(exp, AttributeRef):
                return "%s" % e
            else:
                return "%s as ?%s" % (e, name)
        emitters = ", ".join([formatemitter(name, exp)
                              for (name, exp) in self.emitters])
        return """
  SELECT %s
  {
   %s
  }
  """ % (emitters, self.input.compile())


class SPARQLAlgebra(Algebra):

    def opt_rules(self, **kwargs):
        rules = [
            raco.rules.CrossProduct2Join(),
            raco.rules.SplitSelects(),
            raco.rules.PushSelects(),
            raco.rules.DeDupBroadcastInputs(),
            raco.rules.OneToOne(algebra.Scan, SPARQLScan),
            raco.rules.OneToOne(algebra.Store, SPARQLStore),
            raco.rules.OneToOne(algebra.Select, SPARQLSelect),
            raco.rules.OneToOne(algebra.Apply, SPARQLApply),
            raco.rules.OneToOne(algebra.Join, SPARQLJoin),
        ]

        return rules
