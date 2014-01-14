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

class SimpleDecomposableAggregate(AggregateExpression):
    """An aggregate expression that yields a trivial distibuted decomposition.

    The goal is to split a single logical aggregate into two aggregates.  First,
    a "local" aggregate is applied on each machine.  Next, the data is shuffled
    on the grouping keys, and a second "combiner" aggregate is applied to the
    resulting values.

    We assume that the local aggregate is the same as the logical aggregate.
    The combiner can be arbitrary.

    """

    def get_combiner_class(self):
        """Return the class of the combiner aggregate.

        By default, return the same class as the local aggregate.
        """
        return self.__class__

class MAX(UnaryFunction, SimpleDecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        return max(inputs)

class MIN(UnaryFunction, SimpleDecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        return min(inputs)

class COUNTALL(ZeroaryOperator, SimpleDecomposableAggregate):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        return len(tuple_iterator)

    def get_combiner_class(self):
        return SUM

class COUNT(UnaryFunction, AggregateExpression):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        count = 0
        for t in inputs:
            if t is not None:
                count += 1
        return count

    def get_combiner_class(self):
        return SUM

class SUM(UnaryFunction, SimpleDecomposableAggregate):
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
