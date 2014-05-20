"""
An expression language for Raco: functions, booleans, aggregates, etc.

Most non-trivial operators and functions are in separate files in this package.
"""

from raco.utility import Printable

from abc import ABCMeta, abstractmethod

import logging
LOG = logging.getLogger(__name__)


class Expression(Printable):
    __metaclass__ = ABCMeta
    literals = None

    @classmethod
    def typeof(cls):
        # By default, we don't know the type
        return None

    @classmethod
    def opstr(cls):
        """Return this operator as it would be printed in a string."""
        if not cls.literals:
            return cls.opname()
        return cls.literals[0]

    @abstractmethod
    def evaluate(self, _tuple, scheme, state=None):
        '''Evaluate an expression in the context of a given tuple and schema.

        This is used for unit tests written against the fake database.
        '''

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

    def add_offset(self, offset):
        """Add a constant offset to every positional reference in this tree"""

        for ex in self.walk():
            if isinstance(ex, UnnamedAttributeRef):
                ex.position += offset

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
        return self.__str__()

    def apply(self, f):
        pass

    def walk(self):
        yield self


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
        return self.__str__()

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
        return self.__str__()

    def postorder(self, f):
        for x in self.left.postorder(f):
            yield x
        for x in self.right.postorder(f):
            yield x
        yield f(self)

    def apply(self, f):
        self.left = f(self.left)
        self.right = f(self.right)

    def leftoffset(self, offset):
        """Add a constant offset to all positional references in the left
        subtree"""
        # TODO this is a very weird mechanism. It's really: take all terms that
        # reference the left child and add the offset to them. The
        # implementation is awkward and, is it correct? Elephant x Rhino!
        if isinstance(self.left, BinaryOperator):
            self.left.leftoffset(offset)
            self.right.leftoffset(offset)
        else:
            self.left.add_offset(offset)

    def rightoffset(self, offset):
        """Add a constant offset to all positional references in the right
        subtree"""
        # TODO see leftoffset
        if isinstance(self.right, BinaryOperator):
            self.left.rightoffset(offset)
            self.right.rightoffset(offset)
        else:
            self.right.add_offset(offset)

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

    def __repr__(self):
        return self.__str__()

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

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def typeof(self):
        # TODO: DANGEROUS
        return type(self.value)

    def evaluate(self, _tuple, scheme, state=None):
        return self.value

    def apply(self, f):
        pass


class StringLiteral(Literal):
    def __str__(self):
        return '"{val}"'.format(val=self.value)


class NumericLiteral(Literal):
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


class NamedAttributeRef(AttributeRef):

    def __init__(self, attributename):
        self.name = attributename

    def __repr__(self):
        return "%s" % (self.name)

    def __str__(self):
        return "%s" % (self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name

    def get_position(self, scheme, state_scheme=None):
        return scheme.getPosition(self.name)


class UnnamedAttributeRef(AttributeRef):

    def __init__(self, position):
        self.position = position

    def __repr__(self):
        return "$%s" % (self.position)

    def __str__(self):
        return "$%s" % (self.position)

    def __eq__(self, other):
        return (other.__class__ == self.__class__
                and other.position == self.position)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.position)

    def __cmp__(self, other):
        assert(other.__class__ == self.__class__)
        if self.position < other.position:
            return -1
        elif self.position == other.position:
            return 0
        else:
            return 1

    def get_position(self, scheme, state_scheme=None):
        return self.position


class StateRef(Expression):

    def evaluate(self, _tuple, scheme, state=None):
        return _tuple[self.get_position(scheme, state.scheme)]

    @abstractmethod
    def get_position(self, scheme, state_scheme):
        """Return the position of the referenced attribute in the given
        scheme"""

    def apply(self, f):
        pass


class UnnamedStateAttributeRef(StateRef):

    def __init__(self, position):
        self.position = position

    def __repr__(self):
        return "$%s" % (self.position)

    def __str__(self):
        return "$%s" % (self.position)

    def evaluate(self, _tuple, scheme, state):
        return state.values[self.position]


class NamedStateAttributeRef(StateRef):

    def __init__(self, attributename):
        self.name = attributename

    def __repr__(self):
        return "%s" % (self.name)

    def __str__(self):
        return "%s" % (self.name)

    def evaluate(self, _tuple, scheme, state):
        return state.values[self.get_position(scheme, state.scheme)]

    def get_position(self, scheme, state_scheme):
        return state_scheme.getPosition(self.name)


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


class IDIVIDE(BinaryOperator):
    literals = ["//"]

    def evaluate(self, _tuple, scheme, state=None):
        return int(self.left.evaluate(_tuple, scheme, state) /
                   self.right.evaluate(_tuple, scheme, state))


class TIMES(BinaryOperator):
    literals = ["*"]

    def evaluate(self, _tuple, scheme, state=None):
        return (self.left.evaluate(_tuple, scheme, state) *
                self.right.evaluate(_tuple, scheme, state))


class TYPE(ZeroaryOperator):

    def __init__(self, rtype):
        self.type = rtype

    def typeof(self):
        return self.type

    def evaluate(self, _tuple, scheme, state=None):
        raise Exception("Cannot evaluate this expression operator")


class FLOAT_CAST(UnaryOperator):

    def evaluate(self, _tuple, scheme, state=None):
        return float(self.input.evaluate(_tuple, scheme, state))


class NEG(UnaryOperator):
    literals = ["-"]

    def evaluate(self, _tuple, scheme, state=None):
        return -1 * self.input.evaluate(_tuple, scheme, state)


class Unbox(ZeroaryOperator):

    def __init__(self, relational_expression, field):
        """Initialize an unbox expression.

        relational_expression is a Myrial AST that evaluates to a relation.

        field is an optional column name/index within the relation.  If None,
        the system uses index 0.
        """
        self.relational_expression = relational_expression
        self.field = field

    def evaluate(self, _tuple, scheme, state=None):
        """Raise an error on attempted evaluation.

        Unbox expressions are not "evaluated" in the usual sense.  Rather, they
        are replaced with raw attribute references at evaluation time.
        """
        raise NotImplementedError()


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
        return self.__str__()


import abc


class ExpressionVisitor:
    # TODO: make this more complete for kinds of expressions

    __metaclass__ = abc.ABCMeta

    def visit(self, expr):
        # use expr to dispatch to appropriate visit_* method
        typename = type(expr).__name__
        dispatchTo = getattr(self, "visit_%s" % (typename,))
        return dispatchTo(expr)

    @abc.abstractmethod
    def visit_NOT(self, unaryExpr):
        return

    @abc.abstractmethod
    def visit_AND(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_OR(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_EQ(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_NEQ(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_GT(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_LT(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_GTEQ(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_LTEQ(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_NamedAttributeRef(self, named):
        return

    @abc.abstractmethod
    def visit_UnnamedAttributeRef(self, unnamed):
        return

    @abc.abstractmethod
    def visit_StringLiteral(self, stringLiteral):
        return

    @abc.abstractmethod
    def visit_NumericLiteral(self, numericLiteral):
        return

    @abc.abstractmethod
    def visit_DIVIDE(self, binaryExpr):
        return

    @abc.abstractmethod
    def visit_PLUS(self, binaryExpr):
        return

    @abstractmethod
    def visit_MINUS(self, binaryExpr):
        return

    @abstractmethod
    def visit_IDIVIDE(self, binaryExpr):
        return

    @abstractmethod
    def visit_TIMES(self, binaryExpr):
        return

    @abstractmethod
    def visit_NEG(self, unaryExpr):
        return


class SimpleExpressionVisitor(ExpressionVisitor):
    @abstractmethod
    def visit_unary(self, unaryexpr):
        pass

    @abstractmethod
    def visit_binary(self, binaryexpr):
        pass

    @abstractmethod
    def visit_zeroary(self, zeroaryexpr):
        pass

    @abstractmethod
    def visit_nary(self, naryexpr):
        pass

    def visit_attr(self, attr):
        pass

    def visit_NOT(self, unaryExpr):
        self.visit_unary(unaryExpr)

    def visit_AND(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_OR(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_EQ(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_NEQ(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_GT(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_LT(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_GTEQ(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_LTEQ(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_NamedAttributeRef(self, named):
        self.visit_attr(named)

    def visit_UnnamedAttributeRef(self, unnamed):
        self.visit_attr(unnamed)

    def visit_StringLiteral(self, stringLiteral):
        self.visit_zeroary(stringLiteral)

    def visit_NumericLiteral(self, numericLiteral):
        self.visit_zeroary(numericLiteral)

    def visit_DIVIDE(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_PLUS(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_MINUS(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_IDIVIDE(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_TIMES(self, binaryExpr):
        self.visit_binary(binaryExpr)

    def visit_NEG(self, unaryExpr):
        self.visit_unary(unaryExpr)
