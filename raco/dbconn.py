"""Maintain a connection to a SQL database.

This class translates between raco's internal data types and sqlalchemy's
representation.
"""

import collections

from sqlalchemy import (Column, Table, MetaData, Integer, String, DateTime,
                        Float, Boolean, create_engine, select, text)

from raco.scheme import Scheme
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


class DBConnection(object):

    def __init__(self, connection_string='sqlite:///:memory:', echo=False):
        """Initialize a database connection."""
        self.engine = create_engine(connection_string, echo=echo)
        self.metadata = MetaData()
        self.metadata.bind = self.engine

    def get_scheme(self, rel_key):
        """Return the schema associated with a relation key."""

        table = self.metadata.tables[str(rel_key)]
        return Scheme((c.name, type_to_raco[type(c.type)])
                      for c in table.columns)

    def add_table(self, rel_key, schema, tuples=None):
        """Add a table to the database."""
        self.delete_table(rel_key, ignore_failure=True)
        assert str(rel_key) not in self.metadata.tables

        columns = [Column(n, raco_to_type[t](), nullable=False)
                   for n, t in schema.attributes]
        table = Table(str(rel_key), self.metadata, *columns)
        table.create(self.engine)
        if tuples:
            tuples = [{n: v for n, v in zip(schema.get_names(), tup)}
                      for tup in tuples]
            if tuples:
                self.engine.execute(table.insert(), tuples)

    def append_table(self, rel_key, tuples):
        """Append tuples to an existing relation."""
        scheme = self.get_scheme(rel_key)

        table = self.metadata.tables[str(rel_key)]
        tuples = [{n: v for n, v in zip(scheme.get_names(), tup)}
                  for tup in tuples]
        if tuples:
            self.engine.execute(table.insert(), tuples)

    def num_tuples(self, rel_key):
        """Return number of tuples of rel_key """
        table = self.metadata.tables[str(rel_key)]
        return self.engine.execute(table.count()).scalar()

    def get_table(self, rel_key):
        """Retrieve the contents of a table as a bag (Counter)."""
        table = self.metadata.tables[str(rel_key)]
        s = select([table])
        return collections.Counter(tuple(t) for t in self.engine.execute(s))

    def delete_table(self, rel_key, ignore_failure=False):
        """Delete a table from the database."""
        try:
            table = self.metadata.tables[str(rel_key)]
            table.drop(self.engine)
            self.metadata.remove(table)
        except:
            if not ignore_failure:
                raise

    def get_sql_output(self, sql):
        """Retrieve the result of a query as a bag (Counter)."""
        s = text(sql)
        return collections.Counter(tuple(t) for t in self.engine.execute(s))