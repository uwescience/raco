# coding=utf-8
""" Tests for expressions with UDFs """
from collections import Counter

from python_test import PythonTestCase
from raco.algebra import Apply
from raco.backends.myria.connection import FunctionTypes
from raco.python import convert
from raco.python.exceptions import PythonArgumentException
from raco.types import STRING_TYPE, BOOLEAN_TYPE, LONG_TYPE, INT_TYPE


class TestUDF(PythonTestCase):
    def _execute(self, udfs, query, expected):
        projection = convert(query, [self.schema], udfs)
        self.assertIsNotNone(projection)

        expression = Apply([('out', projection)], self.scan)
        plan = self.get_query(expression)
        return self.check_result(plan, expected)

    @staticmethod
    def _make_udf(name, source, out_type, arity):
        return {'name': name,
                'source': source,
                'outputType': out_type,
                'inputSchema': [INT_TYPE for _ in xrange(arity)],
                'lang': FunctionTypes.PYTHON}

    def test_invocation(self):
        self._execute(
            [self._make_udf('udf', 'lambda i: i == 5', BOOLEAN_TYPE, 1)],
            """lambda t: udf(t[0])""",
            Counter([(0,)] * 6 + [(1,)]))

    def test_invocation_in_expression(self):
        self._execute(
            [self._make_udf('udf', 'lambda i: i == 5', LONG_TYPE, 1)],
            """lambda t: udf(t[0]) + 1""",
            Counter([(1,)] * 6 + [(2,)]))

    def test_multiple_invocation(self):
        self._execute(
            [self._make_udf('udf', 'lambda i: i == 5', LONG_TYPE, 1),
             self._make_udf('udf2', 'lambda i: i < 10', LONG_TYPE, 1)],
            """lambda t: udf(t[0]) + udf2(t[0])""",
            Counter([(1,)] * 6 + [(2,)]))

    def test_unknown_udf(self):
        self.assertRaises(PythonArgumentException,
                          lambda: self._execute(
                              [self._make_udf('udf', 'lambda i: i == 5',
                                              LONG_TYPE, 1)],
                              """lambda t: udffoo(t[0])""",
                              Counter([(1,)] * 6 + [(2,)])))

    def test_arity(self):
        self._execute(
            [self._make_udf('udf', 'lambda i,j: i == j', LONG_TYPE, 2)],
            """lambda t: udf(t[0], t[0])""",
            Counter([(1,)] * len(self.emp_table)))

    def test_string_parameter(self):
        self._execute(
            [self._make_udf('udf', 'lambda s: s[0] == "B"', LONG_TYPE, 1)],
            """lambda t: udf(t[2])""",
            Counter([(0,)] * 6 + [(1,)]))

    def test_string_return(self):
        self._execute(
            [self._make_udf('udf', 'lambda s: s[0]', STRING_TYPE, 1)],
            """lambda t: udf(t[2])""",
            Counter({('D',): 2, ('A',): 1, ('V',): 1, ('B',): 1,
                    ('M',): 1, ('S',): 1}))
