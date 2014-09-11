"""
Aggregate expressions for use in Raco
"""

from .expression import *
from .function import UnaryFunction, SQRT, POW
from .statevar import *
from raco import types
from abc import abstractmethod
import math


class DecomposableAggregateState(object):
    """State associated with decomposable aggregates.

    :param local_emitters: A list of one or more aggregates to run prior to
    the shuffle operation
    :param local_statemods: A list of StateVars associated with local aggs
    :param remote_emitters: A list of one or more aggregates to run after
    the shuffle operation
    :param remote_statemods: A list of StateVars associated with remote aggs
    """
    def __init__(self, local_emitters, local_statemods,
                 remote_emitters, remote_statemods):

        assert all(isinstance(a, UdaAggregateExpression) for a in local_emitters)  # noqa
        assert all(isinstance(a, UdaAggregateExpression) for a in remote_emitters)  # noqa
        assert all(isinstance(s, StateVar) for s in local_statemods)
        assert all(isinstance(s, StateVar) for s in remote_statemods)

        self.local_emitters = local_emitters
        self.local_statemods = local_statemods
        self.remote_emitters = remote_emitters
        self.remote_statemods = remote_statemods

    def get_local_emitters(self):
        return self.local_emitters

    def get_remote_emitters(self):
        return self.remote_emitters

    def get_local_statemods(self):
        return self.local_statemods

    def get_remote_statemods(self):
        return self.remote_statemods


class AggregateExpression(Expression):
    def is_decomposable(self):
        return self.get_decomposable_state() is not None

    @abstractmethod
    def get_decomposable_state(self):
        pass


class BuiltinAggregateExpression(AggregateExpression):
    def evaluate(self, _tuple, scheme, state=None):
        raise NotImplementedError("{expr}.evaluate".format(expr=type(self)))

    def get_decomposable_state(self):
        return None

    @abstractmethod
    def evaluate_aggregate(self, tuple_iterator, scheme):
        """Evaluate an aggregate over a bag of tuples"""


class UdaAggregateExpression(AggregateExpression, UnaryOperator):
    """A user-defined aggregate.

    A UDA wraps an emit expression that is responsible for emitting a
    value for each tuple group.
    """
    def __init__(self, emitter, decomposable_state=None):
        UnaryOperator.__init__(self, emitter)
        self.decomposable_state = decomposable_state

    def evaluate(self, _tuple, scheme, state=None):
        """Evaluate the UDA sub-expression.

        Note that the emitter should only reference the state argument.
        """
        return self.input.evaluate(None, None, state)

    def typeof(self, scheme, state_scheme):
        return self.input.typeof(None, state_scheme)

    def is_decomposable(self):
        return self.decomposable_state is not None

    def get_decomposable_state(self):
        return self.decomposable_state

    def set_decomposable_state(self, ds):
        self.decomposable_state = ds

    def __repr__(self):
        return "{op}({se!r})".format(op=self.opname(), se=self.input)

    def __str__(self):
        return 'UDA(%s)' % self.input


class LocalAggregateOutput(object):
    """Dummy placeholder to refer to the output of a local aggregate."""
    def __init__(self, index=0):
        """Initialize a LocalAggregateOutput

        :param index: The index into the array of local aggregate outputs.
        """
        self.index = index


class MergeAggregateOutput(object):
    """Dummy placeholder to refer to the output of a merge aggregate."""
    def __init__(self, pos):
        """Instantiate a merge aggregate object.

        pos is the position relative to the start of the remote aggregate.
        """
        self.pos = pos

    def to_absolute(self, offsets):
        return UnnamedAttributeRef(offsets[self.pos])


def rebase_local_aggregate_output(expr, offset):
    """Convert LocalAggregateOutput instances to raw column references."""
    assert isinstance(expr, Expression)

    def convert(n):
        if isinstance(n, LocalAggregateOutput):
            return UnnamedAttributeRef(n.index + offset)
        n.apply(convert)
        return n
    return convert(expr)


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
