# coding=utf-8
""" Tests for projection expressions """

from collections import Counter
from raco.python import convert
from raco.algebra import Apply
from python_test import PythonTestCase


class TestProjection(PythonTestCase):
    def _execute_projection(self, query, expected):
        projection = convert(query, [self.schema])
        self.assertIsNotNone(projection)

        expression = Apply([('out', projection)], self.scan)
        plan = self.get_query(expression)
        return self.check_result(plan, expected)

    def test_name(self):
        self._execute_projection("""lambda t: t.name""",
                                 Counter([(t[2],) for t in self.emp_table]))

    def test_expression(self):
        self._execute_projection("""lambda t: t.id + 1""",
                                 Counter([(t[0] + 1,) for t
                                          in self.emp_table]))

    def test_index(self):
        self._execute_projection("""lambda t: t[2]""",
                                 Counter([(t[2],) for t in self.emp_table]))
