import raco.myrial.groupby as groupby
import raco.myrial.multiway as multiway
import raco.algebra
import raco.expression as sexpr
import raco.catalog
import raco.scheme
from raco.language import MyriaAlgebra
from raco.algebra import LogicalAlgebra
from raco.myrialang import compile_to_json
from raco.compile import optimize
from raco import relation_key

import collections
import types
import copy

class DuplicateAliasException(Exception):
    """Bag comprehension arguments must have different alias names."""
    pass

class InvalidStatementException(Exception):
    pass

class NoSuchRelationException(Exception):
    pass

def lookup_symbol(symbols, _id):
    return copy.copy(symbols[_id])

class ExpressionProcessor(object):
    """Convert syntactic expressions into relational algebra operations."""
    def __init__(self, symbols, catalog, use_dummy_schema=False):
        self.symbols = symbols
        self.catalog = catalog
        self.use_dummy_schema = use_dummy_schema

    def evaluate(self, expr):
        method = getattr(self, expr[0].lower())
        return method(*expr[1:])

    def alias(self, _id):
        return lookup_symbol(self.symbols, _id)

    def scan(self, rel_key):
        """Scan a database table."""
        assert isinstance(rel_key, relation_key.RelationKey)
        try:
            scheme = self.catalog.get_scheme(rel_key)
        except KeyError:
            if not self.use_dummy_schema:
                raise NoSuchRelationException(rel_key)

            # Create a dummy schema suitable for emitting plans
            scheme = raco.scheme.DummyScheme()

        return raco.algebra.Scan(rel_key, scheme)

    def table(self, emit_clause):
        """Emit a single-row table literal."""
        emit_args = []
        for clause in emit_clause:
            emit_args.extend(clause.expand({}))

        from_args = collections.OrderedDict()
        from_args['$$SINGLETON$$'] = raco.algebra.SingletonRelation()

        # Add unbox relations to the from_args dictionary
        for name, sexpr in emit_args:
            self.extract_unbox_args(from_args, sexpr)

        op, info = multiway.merge(from_args)

        # rewrite clauses in terms of the new schema
        emit_args = [(name, multiway.rewrite_refs(sexpr, from_args, info))
                      for (name, sexpr) in emit_args]

        return raco.algebra.Apply(emitters=emit_args, input=op)

    @staticmethod
    def empty(_scheme):
        if not _scheme:
            _scheme = raco.scheme.Scheme()
        return raco.algebra.EmptyRelation(_scheme)

    def select(self, args):
        """Evaluate a select-from-where expression."""
        op = self.bagcomp(args.from_, args.where, args.select)
        if args.distinct:
            op = raco.algebra.Distinct(input=op)
        if args.limit is not None:
            op = raco.algebra.Limit(input=op, count=args.limit)

        return op

    def extract_unbox_args(self, from_args, sexpr):
        def extract(sexpr):
            if isinstance(sexpr, raco.expression.Unbox):
                rex = sexpr.relational_expression
                if not rex in from_args:
                    unbox_op = self.evaluate(rex)
                    from_args[rex] = unbox_op
            return 0 # whatever

        # TODO: get rid of stupid list (required to force evaluation)
        l = list(sexpr.postorder(extract))
        return

    def bagcomp(self, from_clause, where_clause, emit_clause):
        """Evaluate a bag comprehsion.

        from_clause: A list of tuples of the form (id, expr).  expr can
        be None, which means "read the value from the symbol table".

        where_clause: An optional scalar expression (raco.expression).

        emit_clause: A list of EmitArg instances, each defining one or more
        output columns.
        """

        # Make sure no aliases were reused: [FROM X, X EMIT *] is illegal
        from_aliases = set([x[0] for x in from_clause])
        if len(from_aliases) != len(from_clause):
            raise DuplicateAliasException();

        # For each FROM argument, create a mapping from ID to operator
        # (id, raco.algebra.Operator)
        from_args = collections.OrderedDict()

        for _id, expr in from_clause:
            if expr:
                from_args[_id] =  self.evaluate(expr)
            else:
                from_args[_id] = lookup_symbol(self.symbols, _id)

        # Expand wildcards into a list of output columns
        assert emit_clause # There should always be something to emit
        emit_args = []
        for clause in emit_clause:
            emit_args.extend(clause.expand(from_args))

        orig_op, _info = multiway.merge(from_args)
        orig_schema_length = len(orig_op.scheme())

        # Add unbox relations to the from_args dictionary
        for name, sexpr in emit_args:
            self.extract_unbox_args(from_args, sexpr)
        if where_clause:
            self.extract_unbox_args(from_args, where_clause)

        # Create a single RA operation that is the cross of all targets
        op, info = multiway.merge(from_args)

        # HACK: calculate unboxed columns as implicit grouping columns,
        # so they can be used in grouping terms.
        new_schema_length = len(op.scheme())
        implicit_group_by_cols = range(orig_schema_length, new_schema_length)

        # rewrite clauses in terms of the new schema
        if where_clause:
            where_clause = multiway.rewrite_refs(where_clause, from_args, info)
            op = raco.algebra.Select(condition=where_clause, input=op)

        emit_args = [(name, multiway.rewrite_refs(sexpr, from_args, info))
                      for (name, sexpr) in emit_args]

        # Apply any grouping operators
        return groupby.groupby(op, emit_args, implicit_group_by_cols)

    def distinct(self, expr):
        op = self.evaluate(expr)
        return raco.algebra.Distinct(input=op)

    def unionall(self, e1, e2):
        left = self.evaluate(e1)
        right = self.evaluate(e2)
        return raco.algebra.UnionAll(left, right)

    def countall(self, expr):
        op = self.evaluate(expr)
        grouping_list = []
        agg_list = [sexpr.COUNTALL()]
        return raco.algebra.GroupBy(grouping_list, agg_list, op)

    def intersect(self, e1, e2):
        left = self.evaluate(e1)
        right = self.evaluate(e2)
        return raco.algebra.Intersection(left, right)

    def diff(self, e1, e2):
        left = self.evaluate(e1)
        right = self.evaluate(e2)
        return raco.algebra.Difference(left, right)

    def limit(self, expr, count):
        op = self.evaluate(expr)
        return raco.algebra.Limit(input=op, count=count)

    def cross(self, left_target, right_target):
        left = self.evaluate(left_target)
        right = self.evaluate(right_target)

        return raco.algebra.CrossProduct(left, right)

    def join(self, left_target, right_target):
        """Convert parser.JoinTarget arguments into a Join operation"""

        left = self.evaluate(left_target.expr)
        right = self.evaluate(right_target.expr)

        assert len(left_target.columns) == len(right_target.columns)

        def get_attribute_ref(column_ref, scheme, offset):
            """Convert a string or int into an attribute ref on the new table"""
            if type(column_ref) == types.IntType:
                index = column_ref
            else:
                index = scheme.getPosition(column_ref)
            return raco.expression.UnnamedAttributeRef(index + offset)

        left_scheme = left.scheme()
        left_refs = [get_attribute_ref(c, left_scheme, 0)
                     for c in left_target.columns]

        right_scheme = right.scheme()
        right_refs = [get_attribute_ref(c, right_scheme, len(left_scheme))
                      for c in right_target.columns]

        join_conditions = [sexpr.EQ(x, y) for x, y in
                           zip(left_refs, right_refs)]

        # Merge the join conditions into a big AND expression

        def andify(x, y):
            """Merge two scalar expressions with an AND"""
            return sexpr.AND(x, y)

        condition = reduce(andify, join_conditions)
        return raco.algebra.Join(condition, left, right)

class StatementProcessor(object):
    '''Evaluate a list of statements'''

    def __init__(self, catalog=None, use_dummy_schema=False):
        # Map from identifiers (aliases) to raco.algebra.Operation instances
        self.symbols = {}

        # A sequence of plans to be executed by the database
        self.output_ops = []

        self.catalog = catalog
        self.ep = ExpressionProcessor(self.symbols, catalog, use_dummy_schema)

        # Unique identifiers for temporary tables created by DUMP operations
        self.dump_output_id = 0

    def evaluate(self, statements):
        '''Evaluate a list of statements'''
        for statement in statements:
            # Switch on the first tuple entry
            method = getattr(self, statement[0].lower())
            method(*statement[1:])

    def __materialize_result(self, _id, expr, op_list):
        '''Materialize an expression as a temporary table.'''
        child_op = self.ep.evaluate(expr)
        store_op = raco.algebra.StoreTemp(_id, child_op)
        op_list.append(store_op)

        # Point future references of this symbol to a scan of the
        # materialized table.
        self.symbols[_id] = raco.algebra.ScanTemp(_id, child_op.scheme())

    def assign(self, _id, expr):
        '''Map a variable to the value of an expression.'''

        # TODO: Apply chaining when it is safe to do so
        # TODO: implement a leaf optimization to avoid duplicate
        # scan/insertions
        self.__materialize_result(_id, expr, self.output_ops)

    def store(self, _id, rel_key):
        assert isinstance(rel_key, relation_key.RelationKey)

        child_op = lookup_symbol(self.symbols, _id)
        op = raco.algebra.Store(rel_key, child_op)
        self.output_ops.append(op)

    def dump(self, _id):
        child_op = lookup_symbol(self.symbols, _id)
        op = raco.algebra.StoreTemp("__OUTPUT%d__" % self.dump_output_id,
                                    child_op)
        self.dump_output_id += 1
        self.output_ops.append(op)

    def get_logical_plan(self):
        """Return an operator representing the logical query plan."""
        return raco.algebra.Sequence(self.output_ops)

    def get_physical_plan(self):
        """Return an operator representing the physical query plan."""

        # TODO: Get rid of the dummy label argument here.
        # Return first (only) plan; strip off dummy label.
        logical_plan = self.get_logical_plan()
        physical_plans = optimize([('root', logical_plan)],
                                  target=MyriaAlgebra,
                                  source=LogicalAlgebra)
        return physical_plans[0][1]

    def get_json(self):
        lp = self.get_logical_plan()
        pps = optimize([('root', lp)], target=MyriaAlgebra,
                       source=LogicalAlgebra)
        # TODO This is not correct. The first argument is the raw query string,
        # not the string representation of the logical plan
        return compile_to_json(str(lp), lp, pps)

    def dowhile(self, statement_list, termination_ex):
        body_ops = []
        for _type, _id, expr in statement_list:
            if _type != 'ASSIGN':
                # TODO: Better error message
                raise InvalidStatementException('%s not allowed in do/while' %
                                                _type.lower())
            self.__materialize_result(_id, expr, body_ops)

        term_op = self.ep.evaluate(termination_ex)
        op = raco.algebra.DoWhile(raco.algebra.Sequence(body_ops), term_op)
        self.output_ops.append(op)
