from raco import algebra
import raco.catalog
import raco.scheme


class FromFileCatalog(raco.catalog.Catalog):
    def __init__(self, catalog):
        self.catalog = catalog

    def get_scheme(self, rel_key):
        string_key = str(rel_key)
        print string_key
        return raco.scheme.Scheme(self.catalog[string_key])

    @classmethod
    def load_from_file(cls, path):
        with open(path) as fh:
            return cls(eval(fh.read()))

    def get_num_servers(self):
        return 1

    def num_tuples(self, rel_key):
        return algebra.DEFAULT_CARDINALITY