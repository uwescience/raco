from collections import Counter
import raco.algebra as algebra
from raco.expression import (AttributeRef, UnnamedAttributeRef,
                             NumericLiteral, StringLiteral,
                             AggregateExpression, COUNTALL, COUNT, AVG, STDEV)
import raco.rules as rules
import raco.types as types


class FlinkGroupBy(rules.Rule):
    """Flink aggregates are required to not change the signature of the input.
    Thus, we must create a stub schema in preparation of an aggregate."""

    def fire(self, op):
        if not isinstance(op, algebra.GroupBy):
            return op

        scheme = op.scheme()
        child_scheme = op.input.scheme()
        op_types = scheme.get_types()
        child_types = child_scheme.get_types()
        if op_types == child_types:
            # GroupBy does not change scheme, so we're fine
            return op

        op_count = Counter(op_types)
        child_count = Counter(child_types)
        if not all(op_count[k] >= child_count[k]
                   for k in op_count.keys() + child_count.keys()):
            # Child scheme is not a subset of op's, no way to do this as yet
            raise NotImplementedError(
                "Fundamentally incompatible schemas: {} and {}"
                .format(scheme, child_scheme)
            )

        if all(op_types[i] == child_types[i] for i in range(len(child_types))):
            # We just need to append some made up data
            refs = [UnnamedAttributeRef(i) for i in range(len(child_types))]
            for t in op_types[len(child_types):]:
                if t == types.STRING_TYPE:
                    refs.append(StringLiteral("blah"))
                elif t == types.LONG_TYPE:
                    refs.append(NumericLiteral(1))
                elif t == types.DOUBLE_TYPE:
                    refs.append(NumericLiteral(1.0))
                else:
                    raise NotImplementedError("handling type {t} in groupby"
                                              .format(t=t))
            type_adjust_apply = algebra.Apply(
                emitters=[(None, r) for r in refs],
                input=op.input)
            op.input = type_adjust_apply
            assert op_types == type_adjust_apply.scheme().get_types()
            return op

        raise NotImplementedError("{} {}".format(op, op.input))