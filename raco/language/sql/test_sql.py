from collections import Counter, defaultdict
from raco.language.sql.test_case import SQLTestCase

import raco.scheme as scheme
import raco.types as types


class TestScheme(SQLTestCase):
    """Test that we can convert Raco schemes to SQL Alchemy schemes"""
    def test_simple_scheme(self):
        sch = scheme.Scheme()
        sch.addAttribute('w', types.FLOAT_TYPE)
        sch.addAttribute('x', types.INT_TYPE)
        sch.addAttribute('y', types.LONG_TYPE)
        sch.addAttribute('z', types.STRING_TYPE)

        self.db.add_table('simple', sch)

        sch2 = self.db.get_scheme('simple')
        self.assertEquals(sch, sch2)


class TestQuery(SQLTestCase):
    """Test actually compiling plans to SQL."""
    def test_catalog(self):
        self.assertEquals(len(self.emp_table),
                          self.db.num_tuples(self.emp_key))

    def test_simple_scan(self):
        query = """x = scan({emp});
        store(x, OUTPUT);""".format(emp=self.emp_key)

        expected = Counter(self.emp_table)
        self.execute(query, expected)

    def test_column_select(self):
        query = """x = scan({emp});
        y = [from x emit $2];
        store(y, OUTPUT);""".format(emp=self.emp_key)

        expected = Counter((x[2],) for x in self.emp_table)
        self.execute(query, expected)

    def test_rename_column_select(self):
        query = """x = scan({emp});
        y = [from x emit $0 as a, $0 as b];
        store(y, OUTPUT);""".format(emp=self.emp_key)

        expected = Counter((x[0], x[0]) for x in self.emp_table)
        self.execute(query, expected)

    def test_count_query(self):
        query = """x = countall(scan({emp}));
        store(x, OUTPUT);""".format(emp=self.emp_key)

        expected = Counter([(len(self.emp_table),)])
        self.execute(query, expected)

    def test_theta_join_query(self):
        query = """x = scan({emp});
        y = scan({emp});
        z = [from x,y where x.salary < y.salary emit x.*];
        store(z, OUTPUT);""".format(emp=self.emp_key)

        expected = Counter((a, b, c, d)
                           for (a, b, c, d) in self.emp_table
                           for (_, _, _, d2) in self.emp_table
                           if d < d2)
        self.execute(query, expected)

    def test_join_query(self):
        query = """x = scan({emp});
        y = scan({emp});
        z = [from x,y where x.salary = y.salary emit count(*) as cnt];
        store(z, OUTPUT);""".format(emp=self.emp_key)

        size = len([1
                    for (_, _, _, d) in self.emp_table
                    for (_, _, _, d2) in self.emp_table
                    if d == d2])
        expected = Counter([(size,)])
        self.execute(query, expected)

    def test_complex_agg_query(self):
        query = """x = scan({emp});
        z = [from x emit dept_id, max(salary) as max_salary];
        store(z, OUTPUT);""".format(emp=self.emp_key)

        d = defaultdict(int)
        for _, dept_id, _, salary in self.emp_table:
            d[dept_id] = max(d[dept_id], salary)
        expected = Counter(d.items())
        self.execute(query, expected)
