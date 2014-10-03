
"""Maintain a connection to a SQL database.

This class translates between raco's internal data types and sqlalchemy's
representation.
"""

import collections

from sqlalchemy import (Column, Table, MetaData, Integer, String,
                        Float, create_engine, select, func)

from raco.scheme import Scheme
import raco.types as types

type_to_raco = {Integer: types.LONG_TYPE,
                String: types.STRING_TYPE,
                Float: types.FLOAT_TYPE}


raco_to_type = {types.LONG_TYPE: Integer,
                types.INT_TYPE: Integer,
                types.STRING_TYPE: String,
                types.FLOAT_TYPE: Float,
                types.DOUBLE_TYPE: Float}


class DBConnection(object):

    def __init__(self, connection_string='sqlite:///:memory:', echo=False):
        """Initialize a SQLLite connection."""
        self.engine = create_engine(connection_string, echo=echo)
        self.metadata = MetaData()
        self.metadata.bind = self.engine

    def get_scheme(self, rel_key):
        """Return the schema associated with a relation key."""

        table = self.metadata.tables[str(rel_key)]
        return Scheme((c.name, type_to_raco[type(c.type)])
                       for c in table.columns)

    def add_table(self, rel_key, schema, tuples=None):
        """Add a table to the SQLLite database."""

        columns = [Column(n, raco_to_type[t](), nullable=False)
                   for n, t in schema.attributes]
        table = Table(str(rel_key), self.metadata, *columns)
        table.create(self.engine)
        if tuples:
            tuples = [{n: v for n, v in zip(schema.get_names(), tup)}
                      for tup in tuples]
            self.engine.execute(table.insert(), tuples)

    def append_table(self, rel_key, tuples):
        """Append tuples to an existing relation."""
        scheme = self.get_scheme(rel_key)

        table = self.metadata.tables[str(rel_key)]
        tuples = [{n: v for n, v in zip(scheme.get_names(), tup)}
                  for tup in tuples]
        self.engine.execute(table.insert(), tuples)

    def num_tuples(self, rel_key):
        """ Return number of tuples of rel_key """
        table = self.metadata.tables[str(rel_key)]
        return self.engine.execute(table.count()).scalar()

    def get_table(self, rel_key):
        """Retrieve the contents of a table as a tuple iterator."""
        table = self.metadata.tables[str(rel_key)]
        s = select([table])
        return (tuple(t) for t in self.engine.execute(s))

    def delete_table(self, rel_key):
        """Delete a table from the database."""

        table = self.metadata.tables[str(rel_key)]
        self.metadata.remove(table)
