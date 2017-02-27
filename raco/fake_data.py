import collections
import raco.scheme as scheme
import raco.types as types

"""This class contains fake data used by several unit tests."""


class FakeData(object):
    emp_table = collections.Counter([
        # id dept_id name salary
        (1, 2, "Bill Howe", 25000),
        (2, 1, "Dan Halperin", 90000),
        (3, 1, "Andrew Whitaker", 5000),
        (4, 2, "Shumo Chu", 5000),
        (5, 1, "Victor Almeida", 25000),
        (6, 3, "Dan Suciu", 90000),
        (7, 1, "Magdalena Balazinska", 25000)])

    emp_schema = scheme.Scheme([("id", types.INT_TYPE),
                                ("dept_id", types.INT_TYPE),
                                ("name", types.STRING_TYPE),
                                ("salary", types.LONG_TYPE)])

    emp_key = "public:adhoc:employee"

    dept_table = collections.Counter([
        (1, "accounting", 5),
        (2, "human resources", 2),
        (3, "engineering", 2),
        (4, "sales", 7)])

    dept_schema = scheme.Scheme([("id", types.LONG_TYPE),
                                 ("name", types.STRING_TYPE),
                                 ("manager", types.LONG_TYPE)])

    dept_key = "public:adhoc:department"

    numbers_table = collections.Counter([
        (1, 3),
        (2, 5),
        (3, -2),
        (16, -4.3)])

    numbers_schema = scheme.Scheme([("id", types.LONG_TYPE),
                                    ("val", types.DOUBLE_TYPE)])

    numbers_key = "public:adhoc:numbers"

    test_function = ("test", "function_text", 1,
                     "id (INT_TYPE), dept_id (INT_TYPE)",
                     "INT_TYPE", "test_body")
