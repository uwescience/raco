"""
Aggregate expressions for use in Raco
"""

from .expression import *
from .function import UnaryFunction, SQRT, POW
from raco import types
from abc import abstractmethod
import math


class AggregateExpression(Expression):
    pass


class BuiltinAggregateExpression(AggregateExpression):
    def evaluate(self, _tuple, scheme, state=None):
        raise NotImplementedError("{expr}.evaluate".format(expr=type(self)))

    @abstractmethod
    def evaluate_aggregate(self, tuple_iterator, scheme):
        """Evaluate an aggregate over a bag of tuples"""


class UdaAggregateExpression(AggregateExpression, ZeroaryOperator):
    """A user-defined aggregate.

    A UDA wraps a sub-expression that is responsible for emitting a
    value for each tuple group.
    """
    def __init__(self, sub_expression):
        self.sub_expression = sub_expression

    def evaluate(self, _tuple, scheme, state=None):
        """Evaluate the UDA sub-expression.

        Note that the sub-expression should only reference the state argument.
        """
        return self.sub_expression.evaluate(None, None, state)

    def typeof(self, scheme, state_scheme):
        return self.sub_expression.typeof(scheme, state_scheme)


class LocalAggregateOutput(object):
    """Dummy placeholder to refer to the output of a local aggregate."""


class MergeAggregateOutput(object):
    """Dummy placeholder to refer to the output of a merge aggregate."""
    def __init__(self, pos):
        """Instantiate a merge aggregate object.

        pos is the position relative to the start of the remote aggregate.
        """
        self.pos = pos

    def to_absolute(self, offsets):
        return UnnamedAttributeRef(offsets[self.pos])


def finalizer_expr_to_absolute(expr, offsets):
    """Convert a finalizer expression to absolute column positions."""

    assert isinstance(expr, Expression)

    def convert(n):
        if isinstance(n, MergeAggregateOutput):
            n = n.to_absolute(offsets)
        n.apply(convert)
        return n
    return convert(expr)


class DecomposableAggregate(BuiltinAggregateExpression):
    """An aggregate expression that yields a distributed execution plan.

    Execution of a decomposable aggregate proceeds in three phases:

    1) Each logical aggregate maps to one or more "local" aggregates that
    are executed on each local machine.
    2) The data is shuffled, and the output of each local aggregate is
    passed to a "merge" aggregate.
    3) The outputs of the merge aggregates are passed to a "finalizer"
    expression, which produces a single output for each of the original logical
    aggregates.

    For example, the AVERAGE aggregate is expressed as:
    Local = [SUM, COUNT]
    Merge = [SUM, SUM]
    Finalize = DIVIDE($0, $1)
    """

    def get_local_aggregates(self):
        """Return a list of local aggregates.

        By default, local aggregates == logical aggregate"""
        return [self]

    def get_merge_aggregates(self):
        """Return a list of merge aggregates.

        By default, apply the same aggregate on the output of the local
        aggregate.
        """
        return [self.__class__(LocalAggregateOutput())]

    def get_finalizer(self):
        """Return a rule for computing the result from the merge aggregates."""
        return None  # by default, use the result from merge aggregate 0


class MAX(UnaryFunction, DecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        return max(inputs)

    def typeof(self, scheme, state_scheme):
        return self.input.typeof(scheme, state_scheme)


class MIN(UnaryFunction, DecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        return min(inputs)

    def typeof(self, scheme, state_scheme):
        return self.input.typeof(scheme, state_scheme)


class COUNTALL(ZeroaryOperator, DecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        return len(tuple_iterator)

    def get_merge_aggregates(self):
        return [SUM(LocalAggregateOutput())]

    def typeof(self, scheme, state_scheme):
        return types.LONG_TYPE


class COUNT(UnaryFunction, DecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        count = 0
        for t in inputs:
            if t is not None:
                count += 1
        return count

    def get_merge_aggregates(self):
        return [SUM(LocalAggregateOutput())]

    def typeof(self, scheme, state_scheme):
        return types.LONG_TYPE


class SUM(UnaryFunction, DecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)

        return sum(x for x in inputs if x is not None)

    def typeof(self, scheme, state_scheme):
        input_type = self.input.typeof(scheme, state_scheme)
        check_is_numeric(input_type)
        return input_type


class AVG(UnaryFunction, DecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        filtered = list(x for x in inputs if x is not None)
        return sum(filtered) / len(filtered)

    def get_local_aggregates(self):
        return [SUM(self.input), COUNT(self.input)]

    def get_merge_aggregates(self):
        return [SUM(LocalAggregateOutput()), SUM(LocalAggregateOutput())]

    def get_finalizer(self):
        # Note: denominator cannot equal zero because groups always have
        # at least one member.
        return DIVIDE(MergeAggregateOutput(0), MergeAggregateOutput(1))

    def typeof(self, scheme, state_scheme):
        input_type = self.input.typeof(scheme, state_scheme)
        check_is_numeric(input_type)
        return types.DOUBLE_TYPE


class STDEV(UnaryFunction, DecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        filtered = [x for x in inputs if x is not None]

        n = len(filtered)
        if n < 2:
            return 0.0

        mean = float(sum(filtered)) / n
        return math.sqrt(sum((a - mean) ** 2 for a in filtered) / n)

    def get_local_aggregates(self):
        return [SUM(self.input), SUM(TIMES(self.input, self.input)),
                COUNT(self.input)]

    def get_merge_aggregates(self):
        return [SUM(LocalAggregateOutput()), SUM(LocalAggregateOutput()),
                SUM(LocalAggregateOutput())]

    def get_finalizer(self):
        # variance(X) = E(X^2) - E(X)^2
        _sum = MergeAggregateOutput(0)
        ssq = MergeAggregateOutput(1)
        count = MergeAggregateOutput(2)

        return SQRT(MINUS(DIVIDE(ssq, count),
                          POW(DIVIDE(_sum, count),
                              NumericLiteral(2))))

    def typeof(self, scheme, state_scheme):
        input_type = self.input.typeof(scheme, state_scheme)
        check_is_numeric(input_type)
        return types.DOUBLE_TYPE
