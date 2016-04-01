from raco.catalog import Catalog
import raco.scheme as scheme
from operator import mul

from raco.types import INT_TYPE, FLOAT_TYPE
from raco.representation import RepresentationProperties

class SparkCatalog(Catalog):


    def __init__(self, connection):
        self.connection = connection
        self.types_dict = {'int64':INT_TYPE, 'int32': INT_TYPE, 'int16': INT_TYPE, 'int': INT_TYPE, 'float64': FLOAT_TYPE, 'float': FLOAT_TYPE}

    def get_scheme(self, rel_key):
        if not self.connection:
            raise RuntimeError(
                "no schema for relation %s because no connection" % rel_key)

        try:
            print rel_key
            df_scheme = self.connection.get_df(rel_key.relation).dtypes
            print scheme.Scheme([(i, self.types_dict[j]) for (i, j) in df_scheme])
            return scheme.Scheme([(i, self.types_dict[j]) for (i, j) in df_scheme])
        except:
            # TODO: pass through other errors.
            raise LookupError('No relation {} in the catalog'.format(rel_key))

    def get_num_servers(self):
        raise NotImplemented

    def num_tuples(self, rel_key):
        if not self.connection:
            raise RuntimeError(
                "no schema for relation %s because no connection" % rel_key)

        try:
            return self.connection.get_df(rel_key.relation).count()
        except:
            # TODO: pass through other errors.
            raise LookupError('No relation {} in the catalog'.format(rel_key))

        if not self.connection:
            raise LookupError(
                "no cardinality of %s because no connection" % rel_key)

    def partitioning(self, rel_key):
        return RepresentationProperties()
