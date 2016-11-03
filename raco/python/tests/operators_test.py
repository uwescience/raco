# coding=utf-8
""" Tests for operators """

from collections import Counter
from raco.python import convert
from raco.algebra import Select
from python_test import PythonTestCase


class TestOperators(PythonTestCase):
    def _execute(self, query, expected):
        predicate = convert(query, [self.schema])
        self.assertIsNotNone(predicate)

        expression = Select(predicate, self.scan)
        plan = self.get_query(expression)
        return self.check_result(plan, expected)

    # Binary operators

    def test_add(self):
        self._execute("""lambda t: t[0] + 1 == 7""",
                      Counter([(6, 3, "Dan Suciu", 90000)]))

    def test_subtract(self):
        self._execute("""lambda t: t[0] - 1 == 5""",
                      Counter([(6, 3, "Dan Suciu", 90000)]))

    def test_multiply(self):
        self._execute("""lambda t: t[0] * 2 == 12""",
                      Counter([(6, 3, "Dan Suciu", 90000)]))

    def test_divide(self):
        self._execute("""lambda t: t[0] / 2 == 3.5""",
                      Counter([(7, 1, 'Magdalena Balazinska', 25000)]))

    def test_integer_divide(self):
        self._execute("""lambda t: t[0] // 2 == 3""",
                      Counter([(6, 3, "Dan Suciu", 90000),
                               (7, 1, 'Magdalena Balazinska', 25000)]))

    def test_modulo(self):
        self._execute("""lambda t: t[0] % 6 == 0""",
                      Counter([(6, 3, "Dan Suciu", 90000)]))

    # Unary operators

    def test_negation(self):
        self._execute("""lambda t: -t[0] == -6""",
                      Counter([(6, 3, "Dan Suciu", 90000)]))

    def test_cast_integer(self):
        self._execute("""lambda t: int(t[0] / 2) == 3""",
                      Counter([(6, 3, "Dan Suciu", 90000),
                               (7, 1, 'Magdalena Balazinska', 25000)]))

    def test_cast_long(self):
        self._execute("""lambda t: long(t[0] / 2) == 3""",
                      Counter([(6, 3, "Dan Suciu", 90000),
                               (7, 1, 'Magdalena Balazinska', 25000)]))

    def test_cast_float(self):
        self._execute("""lambda t: float(t[0] / 2) == 3.0""",
                      Counter([(6, 3, "Dan Suciu", 90000)]))

    def test_cast_string(self):
        self._execute("""lambda t: str(t[0]) == '6'""",
                      Counter([(6, 3, "Dan Suciu", 90000)]))

    def test_cast_boolean(self):
        self._execute("""lambda t: bool(t[0]) == False""",
                      Counter([]))

    def test_substring(self):
        self._execute("""lambda t: t.name[2:4] == 'um'""",
                      Counter([(4, 2, u'Shumo Chu', 5000)]))

    def test_substring_upper(self):
        self._execute("""lambda t: t.name[10:] == 'Balazinska'""",
                      Counter([(7, 1, u'Magdalena Balazinska', 25000)]))

    def test_substring_index(self):
        self._execute("""lambda t: t.name[10] == 'B'""",
                      Counter([(7, 1, u'Magdalena Balazinska', 25000)]))

    def test_substring_lower(self):
        self._execute("""lambda t: t.name[:9] == 'Magdalena'""",
                      Counter([(7, 1, u'Magdalena Balazinska', 25000)]))
