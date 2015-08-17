from raco.catalog import Catalog
import raco.scheme as scheme
from operator import mul

from raco.types import INT_TYPE, FLOAT_TYPE

def parsescidb(result):
    result = result.replace('\n','').replace('),',')\n').replace('[', '').replace(']', '').replace('(','').replace(')','').replace("'",'')
    recs = [line.split(",") for line in result.split("\n")]
    return recs


class SciDBCatalog(Catalog):

    types_dict = {'int64':INT_TYPE, 'int32': INT_TYPE, 'int16': INT_TYPE, 'int': INT_TYPE, 'float64': FLOAT_TYPE, 'float': FLOAT_TYPE}

    def __init__(self, connection):
        self.connection = connection

    def get_scheme(self, rel_key):
        if not self.connection:
            raise RuntimeError(
                "no schema for relation %s because no connection" % rel_key)

        # TODO: Remove this; testing stub
        if rel_key.user == 'SciDB':
            return scheme.Scheme([("i",INT_TYPE), ("j",INT_TYPE), ("value", FLOAT_TYPE)])
        try:
            qattrs = "attributes({})".format(rel_key.relation)
            qdims = "dimensions({})".format(rel_key.relation)
            attrs = parsescidb(self.connection.execute_afl(qattrs))
            dims = parsescidb(self.connection.execute_afl(qdims))
            dimsch = [(rec[0], self.types_dict[rec[-1]]) for rec in dims]
            attrsch = [(rec[0], self.types_dict[rec[1]]) for rec in attrs]
            return scheme.Scheme(dimsch + attrsch)
        except:
            # TODO: pass through other errors.
            raise LookupError('No relation {} in the catalog'.format(rel_key))

    def get_num_servers(self):
        raise NotImplemented

    def num_tuples(self, rel_key):
        if not self.connection:
            raise RuntimeError(
                "no schema for relation %s because no connection" % rel_key)

        # TODO: Remove this; testing stub
        if rel_key.user == 'SciDB':
            return 100

        try:
            qdims = "dimensions({})".format(rel_key.relation)
            dims = parsescidb(self.connection.execute_afl(qdims))
            sizes = [(int(rec[2]) - int(rec[1])) for rec in dims]
            numtuples = reduce(mul, sizes, 1)
        except:
            # TODO: pass through other errors.
            raise LookupError('No relation {} in the catalog'.format(rel_key))

        if not self.connection:
            raise LookupError(
                "no cardinality of %s because no connection" % rel_key)

        return numtuples
