"""
A RACO language to compile expressions to SQL.
"""

from sqlalchemy import (Column, Table, MetaData, Integer, String,
                        Float, Boolean, DateTime, create_engine, select, func)

import raco.algebra as algebra
from raco.catalog import Catalog
import raco.expression as expression
import raco.scheme as scheme
import raco.types as types


type_to_raco = {Integer: types.LONG_TYPE,
                String: types.STRING_TYPE,
                Float: types.FLOAT_TYPE,
                Boolean: types.BOOLEAN_TYPE,
                DateTime: types.DATETIME_TYPE}


raco_to_type = {types.LONG_TYPE: Integer,
                types.INT_TYPE: Integer,
                types.STRING_TYPE: String,
                types.FLOAT_TYPE: Float,
                types.DOUBLE_TYPE: Float,
                types.BOOLEAN_TYPE: Boolean,
                types.DATETIME_TYPE: DateTime}


class SQLCatalog(Catalog):
    def __init__(self, engine=None):
        self.engine = engine
        self.metadata = MetaData()

    @staticmethod
    def get_num_servers():
        """ Return number of servers in myria deployment """
        return 1

    def num_tuples(self, rel_key):
        """ Return number of tuples of rel_key """
        table = self.metadata.tables[str(rel_key)]
        return self.engine.execute(table.count()).scalar()

    def get_scheme(self, rel_key):
        table = self.metadata.tables[str(rel_key)]
        return scheme.Scheme((c.name, type_to_raco[type(c.type)])
                             for c in table.columns)

    def add_table(self, name, schema):
        columns = [Column(n, raco_to_type[t](), nullable=False)
                   for n, t in schema.attributes]
        # Adds the table to the metadata
        Table(name, self.metadata, *columns)

    def add_tuples(self, name, schema, tuples=None):
        table = self.metadata.tables[name]
        table.create(self.engine)
        if tuples:
            tuples = [{n: v for n, v in zip(schema.get_names(), tup)}
                      for tup in tuples]
            self.engine.execute(table.insert(), tuples)

    def _convert_expr(self, cols, expr, input_scheme):
        if isinstance(expr, expression.AttributeRef):
            return self._convert_attribute_ref(cols, expr, input_scheme)
        if isinstance(expr, expression.ZeroaryOperator):
            return self._convert_zeroary_expr(cols, expr, input_scheme)
        if isinstance(expr, expression.UnaryOperator):
            return self._convert_unary_expr(cols, expr, input_scheme)
        if isinstance(expr, expression.BinaryOperator):
            return self._convert_binary_expr(cols, expr, input_scheme)

        raise NotImplementedError("expression {} to sql".format(type(expr)))

    def _convert_attribute_ref(self, cols, expr, input_scheme):
        if isinstance(expr, expression.NamedAttributeRef):
            expr = expression.toUnnamed(expr, input_scheme)

        if isinstance(expr, expression.UnnamedAttributeRef):
            # Not an elif since the first may actually turn into a UARef
            return cols[expr.position]

        raise NotImplementedError("expression {} to sql".format(type(expr)))

    def _convert_zeroary_expr(self, cols, expr, input_scheme):
        if isinstance(expr, expression.COUNTALL):
            return func.count(cols[0])
        if isinstance(expr, expression.Literal):
            return expr.value
        raise NotImplementedError("expression {} to sql".format(type(expr)))

    def _convert_unary_expr(self, cols, expr, input_scheme):
        input = self._convert_expr(cols, expr.input, input_scheme)
        if isinstance(expr, expression.MAX):
            return func.max(input)
        raise NotImplementedError("expression {} to sql".format(type(expr)))

    def _convert_binary_expr(self, cols, expr, input_scheme):
        left = self._convert_expr(cols, expr.left, input_scheme)
        right = self._convert_expr(cols, expr.right, input_scheme)

        if isinstance(expr, expression.AND):
            return left & right
        if isinstance(expr, expression.OR):
            return left | right
        if isinstance(expr, expression.EQ):
            return left == right
        if isinstance(expr, expression.NEQ):
            return left != right
        if isinstance(expr, expression.LT):
            return left < right
        if isinstance(expr, expression.LTEQ):
            return left <= right
        if isinstance(expr, expression.GT):
            return left > right
        if isinstance(expr, expression.GTEQ):
            return left >= right

        raise NotImplementedError("expression {} to sql".format(type(expr)))

    def _get_zeroary_sql(self, plan):
        if isinstance(plan, algebra.Scan):
            if str(plan.relation_key) not in self.metadata.tables:
                self.add_table(str(plan.relation_key), plan.scheme())
            return self.metadata.tables[str(plan.relation_key)].select()
        raise NotImplementedError("convert {op} to sql".format(op=type(plan)))

    def _get_unary_sql(self, plan):
        input = self.get_sql(plan.input).alias("input")
        input_sch = plan.input.scheme()
        cols = list(input.c)

        if isinstance(plan, algebra.Select):
            cond = self._convert_expr(cols, plan.condition, input_sch)
            return input.select(cond)

        elif isinstance(plan, algebra.Apply):
            clause = [self._convert_expr(cols, e, input_sch).label(name)
                      for (name, e) in plan.emitters]
            return select(clause)

        elif isinstance(plan, algebra.GroupBy):
            a = [self._convert_expr(cols, e, input_sch)
                 for e in plan.aggregate_list]
            g = [self._convert_expr(cols, e, input_sch)
                 for e in plan.grouping_list]
            sel = select(g + a)
            if not plan.grouping_list:
                return sel
            return sel.group_by(*g)

        raise NotImplementedError("convert {op} to sql".format(op=type(plan)))

    def _get_binary_sql(self, plan):
        # Use aliases to resolve duplicate names
        left = self.get_sql(plan.left).alias("left")
        right = self.get_sql(plan.right).alias("right")
        all_cols = left.c + right.c
        all_sch = plan.left.scheme() + plan.right.scheme()
        assert len(all_cols) == len(all_sch)

        if isinstance(plan, algebra.CrossProduct):
            out_cols = zip(all_cols, all_sch.get_names())
            return select([col.label(name) for col, name in out_cols])

        elif isinstance(plan, algebra.ProjectingJoin):
            cond = self._convert_expr(all_cols, plan.condition, all_sch)
            return left.join(right, cond)

        raise NotImplementedError("convert {op} to sql".format(op=type(plan)))

    def _get_nary_sql(self, plan):
        raise NotImplementedError("convert {op} to sql".format(op=type(plan)))

    def get_sql(self, plan):
        if isinstance(plan, algebra.ZeroaryOperator):
            return self._get_zeroary_sql(plan)
        elif isinstance(plan, algebra.UnaryOperator):
            return self._get_unary_sql(plan)
        elif isinstance(plan, algebra.BinaryOperator):
            return self._get_binary_sql(plan)
        elif isinstance(plan, algebra.NaryOperator):
            return self._get_nary_sql(plan)
        raise NotImplementedError("convert {op} to sql".format(op=type(plan)))

    def evaluate(self, plan):
        statement = self.get_sql(plan)
        print statement
        return (tuple(t) for t in self.engine.execute(statement))