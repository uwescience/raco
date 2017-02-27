import json
import unittest

from raco.backends.myria import compile_to_json, MyriaStore, MyriaSink
import raco.fakedb
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
import raco.viz
from raco.replace_with_repr import replace_with_repr
from raco import relation_key


class MyrialTestCase(unittest.TestCase):
    def create_db(self):
        return raco.fakedb.FakeDatabase()

    def setUp(self):
        self.db = self.create_db()
        self.parser = parser.Parser()
        self.new_processor()

    def new_processor(self):
        self.processor = interpreter.StatementProcessor(self.db)

    def parse(self, query):
        """Parse a query"""
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)

    def get_plan(self, query, **kwargs):
        """Get the MyriaL query plan for a query"""
        statements = self.parser.parse(query, udas=kwargs.get('udas', None))

        print statements
        self.processor.evaluate(statements)
        if kwargs.get('logical', False):
            p = self.processor.get_logical_plan(**kwargs)
        else:
            p = self.processor.get_physical_plan(**kwargs)
        # verify that we can stringify p
        # TODO verify the string somehow?
        assert str(p)
        # verify that we can convert p to a dot
        # TODO verify the dot somehow?
        raco.viz.get_dot(p)
        # Test repr
        # FIXME: replace_with_repr() is broken for logical ops
        # (__repr__ doesn't persist any constructor args),
        # so only test for physical ops (where __repr__ persists
        # all constructor args).
        if kwargs.get('logical', False):
            return p
        return replace_with_repr(p)

    def get_logical_plan(self, query, **kwargs):
        """Get the logical plan for a MyriaL query"""
        kwargs['logical'] = True
        return self.get_plan(query, **kwargs)

    def get_physical_plan(self, query, **kwargs):
        """Get the physical plan for a MyriaL query"""
        kwargs['logical'] = False
        return self.get_plan(query, **kwargs)

    def execute_query(self, query, test_logical=False, skip_json=False,
                      output='OUTPUT'):
        """Run a test query against the fake database"""
        plan = self.get_plan(query, logical=test_logical)

        if not test_logical and not skip_json:
            # Test that JSON compilation runs without error
            # TODO: verify the JSON output somehow?
            json_string = json.dumps(compile_to_json(
                "some query", "some logical plan", plan, "myrial"))
            assert json_string

        self.db.evaluate(plan)

        return self.db.get_table(output)

    def evaluate_sink_query(self, query):
        """Run a test with query containing sink operator"""
        plan = self.get_plan(query)

        sink_count = [1]     # see https://www.python.org/dev/peps/pep-3104/

        def replace_sink_with_store(_op):
            rel_key = relation_key.RelationKey.from_string(
                "public:adhoc:sink_{}".format(sink_count[0]))
            new_op = _op
            if isinstance(_op, MyriaSink):
                new_op = MyriaStore(
                    plan=_op.input,
                    relation_key=rel_key)
                sink_count[0] += 1
            new_op.apply(replace_sink_with_store)
            return new_op

        new_plan = replace_sink_with_store(plan)
        self.db.evaluate(new_plan)

    def check_result(self, query, expected, test_logical=False,
                     skip_json=False, output='OUTPUT', scheme=None):
        """Execute a test query with an expected output"""
        actual = self.execute_query(query, test_logical, skip_json, output)
        self.assertEquals(actual, expected)

        if scheme:
            self.assertEquals(self.db.get_scheme(output), scheme)

    def check_scheme(self, query, scheme):
        """Execute a test query with an expected output schema."""
        actual = self.execute_query(query)
        self.assertEquals(self.db.get_scheme('OUTPUT'), scheme)
