from raco.catalog import Catalog
import raco.scheme as scheme
from raco.backends.scidb.catalog import SciDBCatalog
from raco.backends.spark.catalog import SparkCatalog
from raco.types import INT_TYPE, FLOAT_TYPE
from raco.representation import RepresentationProperties
from raco.backends.myria.errors import MyriaError

class FederatedCatalog(Catalog):

    def __init__(self, catalogs=[]):
        self.catalogs = catalogs
        self.temp_relations = []

    def add_to_temp_relations(self, name, catalog):
        self.temp_relations.append({'name': name, 'catalog': catalog})

    def get_catalog(self, name):
        for temp in self.temp_relations:
            if name == temp['name']:
                return temp['catalog']

    def _return_first(self, method, rel_key):
        sch = None
        for cat in self.catalogs:
            try:
                response = getattr(cat, method)(rel_key)
                return (cat, response)
            except LookupError:
                continue
            except Exception:
                print 'Relation not present try other catalogs'
                continue

        if not sch:
            raise LookupError("Relation {} not found in any catalogs".format(rel_key))

    def sourceof(self, rel_key):
        cat, sch = self._return_first("get_scheme", rel_key)
        return cat

    def get_scidb_catalog(self):
        for cat in self.catalogs:
            if isinstance(cat, SciDBCatalog):
                return cat
        assert False, "Couldn't find any scidb catalog.."

    def get_spark_catalog(self):
        for cat in self.catalogs:
            if isinstance(cat, SparkCatalog):
                return cat
        assert False, "Couldn't find any spark catalog.."

    def get_num_servers(self):
        raise NotImplemented

    def get_scheme(self, rel_key):
        cat, sch = self._return_first("get_scheme", rel_key)
        return sch

    def num_tuples(self, rel_key):
        return self._return_first("num_tuples", rel_key)

    def partitioning(self, rel_key):
        return RepresentationProperties()
