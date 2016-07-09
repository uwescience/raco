"""
An expression language for Raco: functions, booleans, aggregates, etc.

Most non-trivial operators and functions are in separate files in this package.
"""
from abc import ABCMeta, abstractmethod
import logging

from raco.utility import Printable
from raco import types

LOG = logging.getLogger(__name__)


class TypeSafetyViolation(Exception):
    pass


def check_type(_type, allowed_types):
    if _type not in allowed_types:
        raise TypeSafetyViolation("Type %s not among %s" % (
            _type, allowed_types))


def check_is_numeric(_type):
    check_type(_type, types.NUMERIC_TYPES)


class Expression(Printable):
    __metaclass__ = ABCMeta
    literals = None

    @abstractmethod
    def typeof(self, scheme, state_scheme):
        """Returns a string describing the expression's return type.

        :param scheme: The schema of the relation corresponding to this
        expression
        :param state_scheme: The schema of the state corresponding to this
        expression; this is None except for the StatefulApply operator.
        :return: A string from among types.type_names.
        """

    @classmethod
    def opstr(cls):
        """Return this operator as it would be printed in a string."""
        if not cls.literals:
            return cls.opname()
        return cls.literals[0]

    @abstractmethod
    def evaluate(self, _tuple, scheme, state=None):
        """Evaluate an expression in the context of a given tuple and schema.

        This is used for unit tests written against the fake database.
        """

    @abstractmethod
    def get_children(self):
        """Return a list of child expressions."""

    def __copy__(self):
        raise RuntimeError("Shallow copy not supported for expressions")

    def postorder(self, f):
        """Apply a function to each node in an expression tree.

        The function argument should return a scalar.  The return value
        of postorder is an iterator over these scalars.
        """
        yield f(self)

    def walk(self):
        """Visit the nodes in an expression tree.

        The return value is an iterator over the tree nodes.  The order is
        unspecified.
        """
        yield self

    @abstractmethod
    def apply(self, f):
        """Replace children with the result of a function"""

    def add_offset_by_terms(self, termsToOffset):
        """Add a constant offset to every positional reference in this tree
        using the map of terms to offset.

        This function assumes that every AttributeRef has been labeled with the
        term that it refers to"""

        for ex in self.walk():
            if isinstance(ex, UnnamedAttributeRef):
                offset = termsToOffset[ex.myTerm]
                LOG.debug("adding offset %s to %s", offset, ex)
                ex.position += offset

    def accept(self, visitor):
        """
        Default visitor accept method. Probably does
        not need to be overridden by leaves, but certainly
        by inner tree nodes.
        """
        visitor.visit(self)


class ZeroaryOperator(Expression):

    def __init__(self):
        pass

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "%s" % self.opstr()

    def __repr__(self):
        return "{op}()".format(op=self.opname())

    def apply(self, f):
        pass

    def walk(self):
        yield self

    def get_children(self):
        return []


class UnaryOperator(Expression):

    def __init__(self, input):
        self.input = input

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.input == other.input

    def __hash__(self):
        return hash(self.__class__) + hash(self.input)

    def __str__(self):
        return "%s%s" % (self.opstr(), self.input)

    def __repr__(self):
        return "{op}({inp!r})".format(op=self.opname(), inp=self.input)

    def postorder(self, f):
        for x in self.input.postorder(f):
            yield x
        yield f(self)

    def apply(self, f):
        self.input = f(self.input)

    def walk(self):
        yield self
        for ex in self.input.walk():
            yield ex

    def accept(self, visitor):
        """For post order stateful visitors"""
        self.input.accept(visitor)
        visitor.visit(self)

    def typeof(self, scheme, state_scheme):
        return self.input.typeof(scheme, state_scheme)

    def get_children(self):
        return [self.input]


class BinaryOperator(Expression):

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
            self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash(self.__class__) + hash(self.left) + hash(self.right)

    def __str__(self):
        return "(%s %s %s)" % (self.left, self.opstr(), self.right)

    def __repr__(self):
        return "{op}({l!r}, {r!r})".format(op=self.opname(), l=self.left,
                                           r=self.right)

    def postorder(self, f):
        for x in self.left.postorder(f):
            yield x
        for x in self.right.postorder(f):
            yield x
        yield f(self)

    def apply(self, f):
        self.left = f(self.left)
        self.right = f(self.right)

    def walk(self):
        yield self
        for ex in self.left.walk():
            yield ex
        for ex in self.right.walk():
            yield ex

    def accept(self, visitor):
        """
        For post-order stateful visitors
        """
        self.left.accept(visitor)
        self.right.accept(visitor)
        visitor.visit(self)

    def typeof(self, scheme, state_scheme):
        lt = types.LONG_TYPE
        ft = types.DOUBLE_TYPE
        type_map = {(lt, lt): lt, (lt, ft): ft, (ft, lt): ft, (ft, ft): ft}

        left_type = self.left.typeof(scheme, state_scheme)
        right_type = self.right.typeof(scheme, state_scheme)

        if (left_type, right_type) in type_map:
            return type_map[(left_type, right_type)]
        else:
            raise TypeSafetyViolation("Can't combine %s, %s for %s" % (
                left_type, right_type, self.__class__))

    def get_children(self):
        return [self.left, self.right]


class NaryOperator(Expression):

    def __init__(self, operands):
        self.operands = operands

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        return self.operands == other.operands

    def __hash__(self):
        return hash(self.__class__) + hash(self.operands)

    def __str__(self):
        return "(%s %s)" % \
            (self.opstr(), " ".join([str(i) for i in self.operands]))

    def postorder(self, f):
        for op in self.operands:
            for x in op.postorder(f):
                yield x
        yield f(self)

    def apply(self, f):
        self.operands = [f(op) for op in self.operands]

    def walk(self):
        yield self
        for op in self.operands:
            for ex in op.walk():
                yield ex

    def accept(self, visitor):
        """
        For post-order stateful visitors
        """
        for op in self.operands:
            op.accept(visitor)
        visitor.visit(self)

    def get_children(self):
        return self.operands


class Literal(ZeroaryOperator):

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.value == other.value

    def __hash__(self):
        return hash(self.__class__) + hash(self.value)

    @staticmethod
    def vars():
        return []

    def __str__(self):
        return str(self.value)

    def typeof(self, scheme, state_scheme):
        return types.python_type_map[type(self.value)]

    def evaluate(self, _tuple, scheme, state=None):
        return self.value

    def apply(self, f):
        pass

    def get_children(self):
        return []

    def __repr__(self):
        return "{op}({val!r})".format(op=self.opname(), val=self.value)


class StringLiteral(Literal):

    def __str__(self):
        return '"{val}"'.format(val=self.value)

    def get_val(self):
        return self.value


class NumericLiteral(Literal):
    pass


class BooleanLiteral(Literal):
    pass


class AttributeRef(Expression):

    def evaluate(self, _tuple, scheme, state=None):
        return _tuple[self.get_position(
            scheme, state.scheme if state else None)]

    @abstractmethod
    def get_position(self, scheme, state_scheme=None):
        """Return the position of the referenced attribute in the given
        scheme"""

    def apply(self, f):
        pass

    def walk(self):
        yield self

    def typeof(self, scheme, state_scheme):
        return scheme.getType(self.get_position(scheme, state_scheme))

    def get_children(self):
        return []


class NamedAttributeRef(AttributeRef):

    def __init__(self, attributename):
        self.name = attributename

    def __str__(self):
        return "%s" % (self.name)

    def __repr__(self):
        return "{op}({att!r})".format(op=self.opname(), att=self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.name == other.name

    def get_position(self, scheme, state_scheme=None):
        return scheme.getPosition(self.name)


class UnnamedAttributeRef(AttributeRef):

    def __init__(self, position, debug_info=None):
        self.debug_info = debug_info
        self.position = position

    def __str__(self):
        if not self.debug_info:
            return "${pos}".format(pos=self.position)
        return "{dbg}".format(dbg=self.debug_info)

    def __repr__(self):
        return "{op}({pos!r}, {dbg!r})".format(op=self.opname(),
                                               pos=self.position,
                                               dbg=self.debug_info)

    def __eq__(self, other):
        return (other.__class__ == self.__class__
                and other.position == self.position)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.position)

    def __cmp__(self, other):
        assert other.__class__ == self.__class__
        return cmp(self.position, other.position)

    def get_position(self, scheme, state_scheme=None):
        return self.position


class StateRef(Expression):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_position(self, scheme, state_scheme):
        """Return the position of the referenced attribute in the given
        scheme"""

    def apply(self, f):
        pass

    def get_children(self):
        return []


class UnnamedStateAttributeRef(StateRef):

    def __init__(self, position):
        self.position = position

    def __str__(self):
        return "$%s" % (self.position)

    def __repr__(self):
        return "{op}({pos!r})".format(op=self.opname(), pos=self.position)

    def get_position(self, scheme, state_scheme):
        return self.position

    def evaluate(self, _tuple, scheme, state):
        return state.values[self.position]

    def typeof(self, scheme, state_scheme):
        assert state_scheme is not None
        return state_scheme.getType(self.position)


class NamedStateAttributeRef(StateRef):

    def __init__(self, attributename):
        self.name = attributename

    def __str__(self):
        return "%s" % (self.name)

    def __repr__(self):
        return "{op}({att!r})".format(op=self.opname(), att=self.name)

    def evaluate(self, _tuple, scheme, state):
        return state.values[self.get_position(scheme, state.scheme)]

    def get_position(self, scheme, state_scheme):
        return state_scheme.getPosition(self.name)

    def typeof(self, scheme, state_scheme):
        assert state_scheme is not None
        return state_scheme.getType(self.get_position(scheme, state_scheme))


class UDF(NaryOperator):
    pass


class PLUS(BinaryOperator):
    literals = ["+"]

    def evaluate(self, _tuple, scheme, state=None):
        return (self.left.evaluate(_tuple, scheme, state) +
                self.right.evaluate(_tuple, scheme, state))


class MINUS(BinaryOperator):
    literals = ["-"]

    def evaluate(self, _tuple, scheme, state=None):
        return (self.left.evaluate(_tuple, scheme, state) -
                self.right.evaluate(_tuple, scheme, state))


class DIVIDE(BinaryOperator):
    literals = ["/"]

    def evaluate(self, _tuple, scheme, state=None):
        return (float(self.left.evaluate(_tuple, scheme, state)) /
                self.right.evaluate(_tuple, scheme, state))

    def typeof(self, scheme, state_scheme):
        check_is_numeric(self.left.typeof(scheme, state_scheme))
        check_is_numeric(self.right.typeof(scheme, state_scheme))
        return types.DOUBLE_TYPE


class IDIVIDE(BinaryOperator):
    literals = ["//"]

    def evaluate(self, _tuple, scheme, state=None):
        return int(self.left.evaluate(_tuple, scheme, state) /
                   self.right.evaluate(_tuple, scheme, state))

    def typeof(self, scheme, state_scheme):
        check_is_numeric(self.left.typeof(scheme, state_scheme))
        check_is_numeric(self.right.typeof(scheme, state_scheme))
        return types.LONG_TYPE


class MOD(BinaryOperator):
    literals = ["%"]

    def evaluate(self, _tuple, scheme, state=None):
        return int(self.left.evaluate(_tuple, scheme, state) %
                   self.right.evaluate(_tuple, scheme, state))

    def typeof(self, scheme, state_scheme):
        check_is_numeric(self.left.typeof(scheme, state_scheme))
        check_is_numeric(self.right.typeof(scheme, state_scheme))
        return types.LONG_TYPE


class TIMES(BinaryOperator):
    literals = ["*"]

    def evaluate(self, _tuple, scheme, state=None):
        return (self.left.evaluate(_tuple, scheme, state) *
                self.right.evaluate(_tuple, scheme, state))


class CAST(UnaryOperator):

    def __init__(self, _type, input):
        """Initialize a cast operator.

        @param _type: A string denoting a type; must be from
        types.ALL_TYPES
        """
        assert _type in types.ALL_TYPES
        self._type = _type
        UnaryOperator.__init__(self, input)

    def __str__(self):
        return "%s(%s, %s)" % (self.opstr(), self._type, self.input)

    def __repr__(self):
        return "{op}({t!r}, {inp!r})".format(op=self.opname(), t=self._type,
                                             inp=self.input)

    def evaluate(self, _tuple, scheme, state=None):
        pytype = types.reverse_python_type_map[self.typeof(None, None)]
        return pytype(self.input.evaluate(_tuple, scheme, state))

    def typeof(self, scheme, state_scheme):
        # Note the lack of type-checking here; I didn't want to codify a
        # particular set of casting rules.
        return types.map_type(self._type)


class NEG(UnaryOperator):
    literals = ["-"]

    def evaluate(self, _tuple, scheme, state=None):
        return -1 * self.input.evaluate(_tuple, scheme, state)

    def typeof(self, scheme, state_scheme):
        input_type = self.input.typeof(scheme, state_scheme)
        check_is_numeric(input_type)
        return input_type


class DottedRef(ZeroaryOperator):
    """A DottedRef represents a reference to a column from a given table."""

    def __init__(self, table_alias, field):
        """Initialize an DottedRef expression.

        :param table_alias: The name of a table alias (a string).
        :param field: The column name/index within the relation.
        """
        assert isinstance(table_alias, basestring)
        self.table_alias = table_alias
        self.field = field

    def evaluate(self, _tuple, scheme, state=None):
        """Raise an error on attempted evaluation.

        DottedRef expressions are not "evaluated" in the usual sense.  Rather,
        they are replaced with raw attribute references during compilation.
        """
        raise NotImplementedError()

    def typeof(self, scheme, state_scheme):
        raise NotImplementedError()  # See above comment

    def __repr__(self):
        return "{op}({re!r}, {f!r})".format(
            op=self.opname(), re=self.table_alias, f=self.field)

    def __str__(self):
        return "{op}({re}.{f})".format(
            op=self.opname(), re=self.table_alias, f=self.field)


class Unbox(DottedRef):
    """Unbox expressions act as a DottedRef, but also implicitly add their
    target argument to the FROM clause.
    """
    def __init__(self, table_name, field):
        """Initialize an unbox expression.

        :param table_name: The name of a table (a string).
        :param field: An optional column name/index within the relation.
        If None, the system uses index 0.
        """
        DottedRef.__init__(self, table_name, field or 0)

        # Name == Alias for unbox expressions
        self.table_name = table_name

    def evaluate(self, _tuple, scheme, state=None):
        """Raise an error on attempted evaluation.

        Unbox expressions are not "evaluated" in the usual sense.  Rather, they
        are replaced with raw attribute references during compilation.
        """
        raise NotImplementedError()

    def typeof(self, scheme, state_scheme):
        raise NotImplementedError()  # See above comment

    def __repr__(self):
        return "{op}({re!r}, {f!r})".format(
            op=self.opname(), re=self.table_name, f=self.field)

    def __str__(self):
        return "{op}({re}.{f})".format(
            op=self.opname(), re=self.table_name,
            f=self.field or "$0")


class Case(Expression):

    def __init__(self, when_tuples, else_expr):
        """Initialize a Case expression.

        :param when_tuples: A list of tuples of the form
        (test_expr, result_expr)
        :type when_tuples: List of tuples of (Expression, Expression)
        :param else_expr: An expression to evaluate if no when clause is
        satisfied.
        :type else_expr: Expression
        """
        self.when_tuples = when_tuples
        self.else_expr = else_expr

    def evaluate(self, _tuple, scheme, state=None):
        for test_expr, result_expr in self.when_tuples:
            if test_expr.evaluate(_tuple, scheme, state):
                return result_expr.evaluate(_tuple, scheme, state)
        return self.else_expr.evaluate(_tuple, scheme, state)

    def postorder(self, f):
        for test_expr, result_expr in self.when_tuples:
            for x in test_expr.postorder(f):
                yield x
            for x in result_expr.postorder(f):
                yield x
        for x in self.else_expr.postorder(f):
            yield x
        yield f(self)

    def apply(self, f):
        self.when_tuples = [(f(test), f(result)) for test, result
                            in self.when_tuples]
        self.else_expr = f(self.else_expr)

    def walk(self):
        yield self
        for test_expr, result_expr in self.when_tuples:
            for x in test_expr.walk():
                yield x
            for x in result_expr.walk():
                yield x
        for x in self.else_expr.walk():
            yield x

    def get_children(self):
        test_exprs, result_exprs = zip(*self.when_tuples)
        return list(test_exprs) + list(result_exprs) + [self.else_expr]

    def to_binary(self):
        """Convert n-ary case statements to a binary case statement."""
        assert len(self.when_tuples) > 0
        if len(self.when_tuples) == 1:
            return self
        else:
            new_when_tuples = [self.when_tuples[0]]
            new_else = Case(self.when_tuples[1:], self.else_expr)
            return Case(new_when_tuples, new_else)

    def __str__(self):
        when_strs = ['WHEN %s THEN %s' % (test, result)
                     for test, result in self.when_tuples]
        return "CASE(%s ELSE %s)" % (' '.join(when_strs), self.else_expr)

    def __repr__(self):
        return "{op}({wt!r}, {els!r})".format(op=self.opname(),
                                              wt=self.when_tuples,
                                              els=self.else_expr)

    def typeof(self, scheme, state_scheme):
        all_exprs = [res_expr for test_expr, res_expr in self.when_tuples]
        all_exprs.append(self.else_expr)
        types = [ex.typeof(scheme, state_scheme) for ex in all_exprs]
        if len(set(types)) != 1:
            raise TypeSafetyViolation(
                "Case expresssions must resolve to a single type")
        return types[0]

    def accept(self, visitor):
        """
        For post-order stateful visitors
        """
        for w in self.when_tuples:
            w[0].accept(visitor)
            w[1].accept(visitor)

        self.else_expr.accept(visitor)
        visitor.visit(self)
