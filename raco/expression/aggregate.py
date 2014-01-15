"""
Aggregate expressions for use in Raco
"""

from .expression import Expression, ZeroaryOperator
from .function import UnaryFunction

from abc import abstractmethod
import math

class AggregateExpression(Expression):
    def evaluate(self, _tuple, scheme):
        """Stub evaluate function for aggregate expressions.

        Aggregate functions do not evaluate individual tuples; rather they
        operate on collections of tuples in the evaluate_aggregate function.
        We return a dummy string so that all tuples containing this aggregate
        hash to the same value.
        """
        return self.opname()

    @abstractmethod
    def evaluate_aggregate(self, tuple_iterator, scheme):
        """Evaluate an aggregate over a bag of tuples"""

class LocalAggregateOutput(object):
    """Dummy placeholder to refer to the output of a local aggregate."""

class DecomposableAggregate(AggregateExpression):
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

    def get_finalize_expression(self):
        """Return a rule for extracting the result from the merge aggregats."""
        return None # use the result from merge aggregate 0

class MAX(UnaryFunction, DecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        return max(inputs)

class MIN(UnaryFunction, DecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        return min(inputs)

class COUNTALL(ZeroaryOperator, DecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        return len(tuple_iterator)

    def get_merge_aggregates(self):
        return [SUM(LocalAggregateOutput())]

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

class SUM(UnaryFunction, DecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)

        sum = 0
        for t in inputs:
            if t is not None:
                sum += t
        return sum

class AVERAGE(UnaryFunction, AggregateExpression):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        filtered = (x for x in inputs if x is not None)

        sum = 0
        count = 0
        for t in filtered:
            sum += t
            count += 1
        return sum / count

class STDEV(UnaryFunction, AggregateExpression):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        filtered = [x for x in inputs if x is not None]

        n = len(filtered)
        if (n < 2):
            return 0.0

        mean = float(sum(filtered)) / n

        std = 0.0
        for a in filtered:
            std = std + (a - mean)**2
        std = math.sqrt(std / n)
        return std
