from raco.catalog import Catalog
import raco.scheme as scheme
from operator import mul


def parsescidb(result):
    lines = result.split("\n")
    recs = [line.split(",") for line in result.split("\n")]
    return recs


class SciDBCatalog(Catalog):

    def __init__(self, connection):
        self.connection = connection

    def get_scheme(self, rel_key):
        if not self.connection:
            raise RuntimeError(
                "no schema for relation %s because no connection" % rel_key)

        try:
            qattrs = "attributes({})".format(rel_key.relation)
            qdims = "dimensions({})".format(rel_key.relation)
            attrs = parsescidb(self.connection.query(qattrs))
            dims = parsescidb(self.connection.query(qdims))
        except:
            # TODO: pass through other errors.
            raise LookupError('No relation {} in the catalog'.format(rel_key))
        dimsch = [(rec[0], rec[-1]) for rec in dims]
        attrsch = [(rec[0], rec[1]) for rec in attrs]
        return scheme.Scheme(dimsch + attrsch)

    def get_num_servers(self):
        raise NotImplemented

    def num_tuples(self, rel_key):
        if not self.connection:
            raise RuntimeError(
                "no schema for relation %s because no connection" % rel_key)

        try:
            qdims = "dimensions({})".format(rel_key.relation)
            dims = parsescidb(self.connection.query(qdims))
            sizes = [(int(rec[2]) - int(rec[1])) for rec in dims]
            numtuples = reduce(operator.mul, sizes, 1)
        except:
            # TODO: pass through other errors.
            raise LookupError('No relation {} in the catalog'.format(rel_key))

        if not self.connection:
            raise LookupError(
                "no cardinality of %s because no connection" % rel_key)

        return numtuples
