# coding=utf-8
""" Tests for predicates and comparators """

from collections import Counter
from raco.python import convert
from raco.algebra import Select
from python_test import PythonTestCase


class TestBooleans(PythonTestCase):
    def _execute_predicate(self, query, expected):
        predicate = convert(query, [self.schema])
        self.assertIsNotNone(predicate)

        expression = Select(predicate, self.scan)
        plan = self.get_query(expression)
        return self.check_result(plan, expected)

    def test_integer_equality(self):
        self._execute_predicate("""lambda t: t[0] == 6""",
                                Counter([(6, 3, "Dan Suciu", 90000)]))

    def test_float_equality(self):
        self._execute_predicate("""lambda t: t[0] == 6.0""",
                                Counter([(6, 3, "Dan Suciu", 90000)]))

    def test_string_equality(self):
        self._execute_predicate("""lambda t: t[2] == 'Dan Suciu'""",
                                Counter([(6, 3, "Dan Suciu", 90000)]))

    def test_inequality(self):
        self._execute_predicate("""lambda t: t[0] != 999""",
                                self.emp_table)

    def test_less_than(self):
        self._execute_predicate("""lambda t: t[0] < 999""",
                                self.emp_table)

    def test_less_than_equal(self):
        self._execute_predicate("""lambda t: t[0] <= 7""",
                                self.emp_table)

    def test_greater_than(self):
        self._execute_predicate("""lambda t: t[0] >= 1""",
                                self.emp_table)

    def test_greater_than_equal(self):
        self._execute_predicate("""lambda t: t[0] >= 7""",
                                Counter([(7, 1, 'Magdalena Balazinska',
                                          25000)]))

    def test_conjunction(self):
        self._execute_predicate(
            """lambda t: t[0] >= 7 and t[2] != 'Magdalena Balazinska'""",
            Counter([]))

    def test_extended_conjunction(self):
        self._execute_predicate(
            """lambda t: t[0] >= 7 and t[1] != 1 and """ +
            """                        t[2] == 'Magdalena Balazinska'""",
            Counter([]))

    def test_disjunction(self):
        self._execute_predicate(
            """lambda t: t[0] >= 999 or t[2] == 'Magdalena Balazinska'""",
            Counter([(7, 1, 'Magdalena Balazinska', 25000)]))

    def test_extended_disjunction(self):
        self._execute_predicate(
            """lambda t: t[0] >= 999 or """ +
            """          t[2] == 'Magdalena Balazinska' or 1 == 1""",
            self.emp_table)

    def test_negation(self):
        self._execute_predicate("""lambda t: not (1 == 1)""",
                                Counter([]))

    def test_mixed_clauses(self):
        self._execute_predicate(
            """lambda t: t[0] >= 0 and """ +
            """          t[2] == 'Magdalena Balazinska' or 1 == 1""",
            self.emp_table)

    def test_true(self):
        self._execute_predicate("""lambda t: True""",
                                self.emp_table)

    def test_false(self):
        self._execute_predicate("""lambda t: False""",
                                Counter([]))
