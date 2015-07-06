
from raco import relation_key
from raco.catalog import Catalog
from raco.algebra import DEFAULT_CARDINALITY
import csv


class ClangTestDatabase(Catalog):
    """
    Interface for table metadata and ingest
    for raco.cpp query processor
    """

    def __init__(self):
        # Map from relation keys to tuples of (Bag, scheme.Scheme)
        self.tables = {}

    def get_num_servers(self):
        return 1

    def num_tuples(self, rel_key):
        return DEFAULT_CARDINALITY

    def ingest(self, rel_key, contents, scheme):
        '''Directly load raw data into the database'''
        if isinstance(rel_key, basestring):
            rel_key = relation_key.RelationKey.from_string(rel_key)
        assert isinstance(rel_key, relation_key.RelationKey)

        with open(rel_key.relation, 'w') as writetable:
            writer = csv.writer(writetable, delimiter=' ')
            for tup in contents:
                writer.writerow(tup)

        self.tables[rel_key] = scheme

    def get_scheme(self, rel_key):
        if isinstance(rel_key, basestring):
            rel_key = relation_key.RelationKey.from_string(rel_key)

        assert isinstance(rel_key, relation_key.RelationKey)

        scheme = self.tables[rel_key]
        return scheme
