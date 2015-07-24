from raco.catalog import Catalog
import raco.scheme as scheme

class FederatedCatalog(Catalog):

    def __init__(self, catalogs=[]):
        self.catalogs = catalogs

    def _return_first(self, method, rel_key):
        sch = None
        for cat in self.catalogs:
            try:
                response = getattr(cat, method)(rel_key)
                return (cat, response)
            except LookupError:
                continue

        if not sch:
            raise LookupError("Relation {} not found in any catalogs".format(rel_key))

    def sourceof(self, rel_key):
        cat, sch = self._return_first("get_scheme", rel_key)
        return cat

    def get_num_servers(self):
        raise NotImplemented

    def get_scheme(self, rel_key):
        cat, sch = self._return_first("get_scheme", rel_key)
        return sch

    def num_tuples(self, rel_key):
        return self._return_first("num_tuples", rel_key)
