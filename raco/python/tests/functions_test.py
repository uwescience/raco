# coding=utf-8
""" Tests for built-in functions"""

from collections import Counter
from raco.python import convert
from raco.algebra import Apply
from python_test import PythonTestCase


class TestFunctions(PythonTestCase):
    def _execute(self, query, expected):
        projection = convert(query, [self.schema])
        self.assertIsNotNone(projection)

        expression = Apply([('out', projection)], self.scan)
        plan = self.get_query(expression)
        return self.check_result(plan, expected)

    def test_worker_id(self):
        self._execute("""lambda t: workerid()""",
                      Counter([(0,)] * len(self.emp_table)))

    def test_random(self):
        self._execute("""lambda t: random() <= 1""",
                      Counter([(True,)] * len(self.emp_table)))

    def test_abs(self):
        self._execute("""lambda t: abs(-t[0]) <= 0""",
                      Counter([(False,)] * len(self.emp_table)))

    def test_fabs(self):
        self._execute("""lambda t: fabs(-t[0]) <= 0.25""",
                      Counter([(False,)] * len(self.emp_table)))

    def test_ceil(self):
        self._execute("""lambda t: ceil(t[0] / 2)""",
                      Counter([(1,), (1,), (2,), (2,), (3,), (3,), (4,)]))

    def test_cosine(self):
        self._execute("""lambda t: cos(0)""",
                      Counter([(1,)] * len(self.emp_table)))

    def test_floor(self):
        self._execute("""lambda t: floor(5.5)""",
                      Counter([(5,)] * len(self.emp_table)))

    def test_log(self):
        self._execute("""lambda t: log(1)""",
                      Counter([(0.0,)] * len(self.emp_table)))

    def test_sine(self):
        self._execute("""lambda t: sin(0)""",
                      Counter([(0.0,)] * len(self.emp_table)))

    def test_square_root(self):
        self._execute("""lambda t: sqrt(4)""",
                      Counter([(2,)] * len(self.emp_table)))

    def test_tangent(self):
        self._execute("""lambda t: tan(0)""",
                      Counter([(0,)] * len(self.emp_table)))

    def test_md5_hash(self):
        self._execute("""lambda t: len(str(md5(t[2])))""",
                      Counter([(19,)] * 6 + [(18,)]))

    def test_length(self):
        self._execute("""lambda t: len(t[2])""",
                      Counter([(9,), (9,), (9,), (12,), (14,), (15,), (20,)]))

    def test_power(self):
        self._execute("""lambda t: pow(2, 3)""",
                      Counter([(8,)] * len(self.emp_table)))

    def test_minimum(self):
        self._execute("""lambda t: min(5, 4)""",
                      Counter([(4,)] * len(self.emp_table)))

    def test_minimum_doubles(self):
        self._execute("""lambda t: min(5.5, 4.5)""",
                      Counter([(4.5,)] * len(self.emp_table)))

    def test_maximum(self):
        self._execute("""lambda t: max(10, 20)""",
                      Counter([(20,)] * len(self.emp_table)))

    def test_maximum_doubles(self):
        self._execute("""lambda t: max(10.5, 20.5)""",
                      Counter([(20.5,)] * len(self.emp_table)))
