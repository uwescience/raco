"""
Aggregate expressions for use in Raco
"""

from .expression import Expression, ZeroaryOperator
from .function import UnaryFunction

from abc import abstractmethod
import math

class AggregateExpression(Expression):
    def evaluate(self, _tuple, scheme, state=None):
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

class MAX(UnaryFunction, AggregateExpression):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        return max(inputs)

class MIN(UnaryFunction, AggregateExpression):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        return min(inputs)

class COUNTALL(ZeroaryOperator, AggregateExpression):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        return len(tuple_iterator)

class COUNT(UnaryFunction, AggregateExpression):
    def evaluate_aggregate(self, tuple_iterator, scheme):
        inputs = (self.input.evaluate(t, scheme) for t in tuple_iterator)
        count = 0
        for t in inputs:
            if t is not None:
                count += 1
        return count

class SUM(UnaryFunction, AggregateExpression):
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
