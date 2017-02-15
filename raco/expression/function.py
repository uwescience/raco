"""
Functions (unary and binary) for use in Raco.
"""

import math
import md5
import random

from raco.expression.udf import Function

from .expression import (ZeroaryOperator, UnaryOperator, BinaryOperator,
                         NaryOperator, types, check_is_numeric, check_type,
                         TypeSafetyViolation, UnnamedAttributeRef)


class UnaryFunction(UnaryOperator):

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.input)

    def __repr__(self):
        return "{op}({inp!r})".format(op=self.opname(), inp=self.input)


class BinaryFunction(BinaryOperator):

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.left, self.right)

    def __repr__(self):
        return "{op}({l!r}, {r!r})".format(op=self.opname(), l=self.left,
                                           r=self.right)


class CustomBinaryFunction(BinaryFunction):

    def __init__(self, name, typ, left, right):
        self.name = name
        self.typ = typ
        super(CustomBinaryFunction, self).__init__(left, right)

    def __repr__(self):
        return "{op}({n!r}, {t!r}, {l!r}, {r!r})".format(op=self.opname(),
                                                         n=self.name,
                                                         t=self.typ,
                                                         l=self.left,
                                                         r=self.right)

    def typeof(self, scheme, state_scheme):
        return self.typ

    literals = []

    def evaluate(self, _tuple, scheme, state=None):
        raise NotImplementedError("Not intended for evaluation")


class ZeroaryFunction(ZeroaryOperator):

    def __str__(self):
        return "%s()" % (self.__class__.__name__)

    def __repr__(self):
        return "{op}()".format(op=self.opname())


class CustomZeroaryFunction(ZeroaryFunction):

    def __init__(self, name, typ):
        self.name = name
        self.typ = typ
        super(CustomZeroaryFunction, self).__init__()

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.name, self.typ)

    def __repr__(self):
        return "{op}({n!r}, {t!r})".format(op=self.opname(),
                                           n=self.name,
                                           t=self.typ)

    def typeof(self, scheme, state_scheme):
        return self.typ

    def evaluate(self, _tuple, scheme, state=None):
        raise NotImplementedError("Not intended for evaluation")


class NaryFunction(NaryOperator):

    def __str__(self):
        return "%s(%s)" % \
            (self.__class__.__name__,
             ",".join([str(op) for op in self.operands]))

    def __repr__(self):
        return "{op}({ch!r})".format(op=self.opname(), ch=self.operands)


class WORKERID(ZeroaryOperator):

    def __str__(self):
        return "%s" % self.__class__.__name__

    def __repr__(self):
        return "{op}()".format(op=self.opname())

    def evaluate(self, _tuple, scheme, state=None):
        return 0

    def typeof(self, scheme, state_scheme):
        return types.LONG_TYPE


class RANDOM(ZeroaryOperator):

    def __str__(self):
        return "%s" % self.__class__.__name__

    def __repr__(self):
        return "{op}()".format(op=self.opname())

    def evaluate(self, _tuple, scheme, state=None):
        return random.random()

    def typeof(self, scheme, state_scheme):
        return types.DOUBLE_TYPE


class UnaryDateToNumFunction(UnaryFunction):
    def evaluate(self, _tuple, scheme, state=None):
        raise NotImplementedError()

    def typeof(self, scheme, state_scheme):
        if self.input.typeof(scheme, state_scheme) != types.STRING_TYPE:
            raise TypeSafetyViolation("Dates can only be strings")
        return types.LONG_TYPE


class YEAR(UnaryDateToNumFunction):
    pass


class MONTH(UnaryDateToNumFunction):
    pass


class DAY(UnaryDateToNumFunction):
    pass


class UnaryDoubleFunction(UnaryFunction):

    """A unary function that returns a double."""

    def typeof(self, scheme, state_scheme):
        input_type = self.input.typeof(scheme, state_scheme)
        check_is_numeric(input_type)
        return types.DOUBLE_TYPE


class UnaryTypePreservingFunction(UnaryFunction):

    def typeof(self, scheme, state_scheme):
        input_type = self.input.typeof(scheme, state_scheme)
        check_is_numeric(input_type)
        return input_type


class VariadicFunction(object):
    def __init__(self, ftype, name, typ, **kwargs):
        self.ftype = ftype
        self.name = name
        self.typ = typ
        self.kwargs = kwargs

    def bind(self, *args):
        return Function(
            ['arg_{}'.format(i) for i in xrange(len(args))],
            self.ftype(self.name,
                       self.typ,
                       *[UnnamedAttributeRef(i) for i in xrange(len(args))],
                       **self.kwargs))


class ABS(UnaryTypePreservingFunction):

    def evaluate(self, _tuple, scheme, state=None):
        return abs(self.input.evaluate(_tuple, scheme, state))


class CEIL(UnaryDoubleFunction):

    def evaluate(self, _tuple, scheme, state=None):
        return math.ceil(self.input.evaluate(_tuple, scheme, state))


class COS(UnaryDoubleFunction):

    def evaluate(self, _tuple, scheme, state=None):
        return math.cos(self.input.evaluate(_tuple, scheme, state))


class FLOOR(UnaryDoubleFunction):

    def evaluate(self, _tuple, scheme, state=None):
        return math.floor(self.input.evaluate(_tuple, scheme, state))


class LOG(UnaryDoubleFunction):

    def evaluate(self, _tuple, scheme, state=None):
        return math.log(self.input.evaluate(_tuple, scheme, state))


class SIN(UnaryDoubleFunction):

    def evaluate(self, _tuple, scheme, state=None):
        return math.sin(self.input.evaluate(_tuple, scheme, state))


class SQRT(UnaryDoubleFunction):

    def evaluate(self, _tuple, scheme, state=None):
        return math.sqrt(self.input.evaluate(_tuple, scheme, state))


class TAN(UnaryDoubleFunction):

    def evaluate(self, _tuple, scheme, state=None):
        return math.tan(self.input.evaluate(_tuple, scheme, state))


class MD5(UnaryFunction):

    def typeof(self, scheme, state_scheme):
        return types.LONG_TYPE

    def evaluate(self, _tuple, scheme, state=None):
        """Preserve 64 bits of the md5 hash function."""
        m = md5.new()
        m.update(str(self.input.evaluate(_tuple, scheme, state)))
        return int(m.hexdigest(), 16) >> 64


class POW(BinaryFunction):
    literals = ['POW']

    def evaluate(self, _tuple, scheme, state=None):
        return pow(self.left.evaluate(_tuple, scheme, state),
                   self.right.evaluate(_tuple, scheme, state))

    def typeof(self, scheme, state_scheme):
        lt = self.left.typeof(scheme, state_scheme)
        check_is_numeric(lt)
        rt = self.right.typeof(scheme, state_scheme)
        check_is_numeric(rt)

        return types.DOUBLE_TYPE


class CompareFunction(BinaryFunction):

    def typeof(self, scheme, state_scheme):
        lt = self.left.typeof(scheme, state_scheme)
        rt = self.right.typeof(scheme, state_scheme)
        if lt != rt:
            raise TypeSafetyViolation("Can't compare %s with %s" % (
                lt, rt))
        return lt


class LESSER(CompareFunction):
    literals = ['LESSER']

    def evaluate(self, _tuple, scheme, state=None):
        return min(self.left.evaluate(_tuple, scheme, state),
                   self.right.evaluate(_tuple, scheme, state))


class GREATER(CompareFunction):
    literals = ['GREATER']

    def evaluate(self, _tuple, scheme, state=None):
        return max(self.left.evaluate(_tuple, scheme, state),
                   self.right.evaluate(_tuple, scheme, state))


class SUBSTR(NaryFunction):
    literals = ["SUBSTR"]

    def evaluate(self, _tuple, scheme, state=None):
        inputStr = self.operands[0].evaluate(_tuple, scheme, state)
        beginIdx = self.operands[1].evaluate(_tuple, scheme, state)
        endIdx = self.operands[2].evaluate(_tuple, scheme, state)
        return inputStr[beginIdx:endIdx]

    def typeof(self, scheme, state_scheme):
        check_type(self.operands[0].typeof(scheme, state_scheme), types.STRING_TYPE)  # noqa
        check_type(self.operands[1].typeof(scheme, state_scheme), types.LONG_TYPE)  # noqa
        check_type(self.operands[2].typeof(scheme, state_scheme), types.LONG_TYPE)  # noqa

        return types.STRING_TYPE


class LEN(UnaryFunction):
    literals = ["LEN"]

    def evaluate(self, _tuple, scheme, state=None):
        return len(self.input.evaluate(_tuple, scheme, state))

    def typeof(self, scheme, state_scheme):
        input_type = self.input.typeof(scheme, state_scheme)
        if input_type != types.STRING_TYPE:
            raise TypeSafetyViolation("Must be a string for %s" % (
                self.__class__,))
        return types.LONG_TYPE


class PythonUDF(NaryFunction):

    literals = []

    def __init__(self, name, typ, *args, **kwargs):
        super(PythonUDF, self).__init__(args)
        self.name = name
        self.source = kwargs.get('source', None)
        self.func = eval(self.source) if self.source else None
        self.typ = typ
        self.arguments = tuple(args)

    def __str__(self):
        return "%s(%s, %s, %s)" % (self.__class__.__name__,
                                   self.name,
                                   map(str, self.arguments),
                                   self.typ)

    def __repr__(self):
        return "{op}({n!r},{t!r},*{a!r}, source={s!r})".format(
            op=self.opname(),
            n=self.name,
            a=self.arguments,
            t=self.typ,
            s=self.source)

    def typeof(self, scheme, state_scheme):
        return self.typ

    def set_typ(self, typ):
        self.typ = typ

    def apply(self, f):
        map(lambda a: f(a), self.arguments)

    def evaluate(self, _tuple, scheme, state=None):
        if self.func:
            return self.func(*map(lambda a: a.evaluate(_tuple, scheme, state),
                                  self.arguments))
        else:
            raise NotImplementedError()


class SPLIT(BinaryFunction):
    literals = ["SPLIT"]

    # TODO: implement once fakedb can handle flatmap functions
    def evaluate(self, _tuple, scheme, state=None):
        raise NotImplementedError()

    def typeof(self, scheme, state_scheme):
        lt = self.left.typeof(scheme, state_scheme)
        check_type(lt, types.STRING_TYPE)
        rt = self.right.typeof(scheme, state_scheme)
        check_type(rt, types.STRING_TYPE)
        return types.STRING_TYPE


class SEQUENCE(UnaryFunction):
    literals = ["SEQUENCE"]

    # TODO: implement once fakedb can handle flatmap functions
    def evaluate(self, _tuple, scheme, state=None):
        raise NotImplementedError()

    def typeof(self, scheme, state_scheme):
        input_type = self.input.typeof(scheme, state_scheme)
        if input_type != types.LONG_TYPE:
            raise TypeSafetyViolation("Must be an integer for %s" % (
                self.__class__,))
        return types.LONG_TYPE


class NGRAM(BinaryFunction):
    literals = ["NGRAM"]

    # TODO: implement once fakedb can handle flatmap functions
    def evaluate(self, _tuple, scheme, state=None):
        raise NotImplementedError()

    def typeof(self, scheme, state_scheme):
        lt = self.left.typeof(scheme, state_scheme)
        check_type(lt, types.STRING_TYPE)
        rt = self.right.typeof(scheme, state_scheme)
        check_type(rt, types.LONG_TYPE)
        return types.STRING_TYPE
