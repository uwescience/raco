from raco.catalog import Catalog
import raco.scheme as scheme
from raco.representation import RepresentationProperties
from raco.expression import UnnamedAttributeRef as AttIndex
from raco.catalog import DEFAULT_CARDINALITY

class AccumuloCatalog(Catalog):

    def __init__(self, connection):
        self.connection = connection

    def get_scheme(self, rel_key):
        accumulo_rel_key = '{}_{}_{}'.format(rel_key.user, rel_key.program, rel_key.relation)

        if not self.connection:
            raise RuntimeError(
                "no schema for relation %s because no connection" % rel_key)

        props = self.connection.getTableProperties(accumulo_rel_key)
            #self.connection.client.getTableProperties(self.connection.self.login, accumulo_rel_key)
        print props

        # todo return list of tuples and types
        # props.scheme
        return scheme.Scheme(  )

    def get_num_servers(self):
        # todo
        return 1

    def num_tuples(self, rel_key):
        return DEFAULT_CARDINALITY

    def partitioning(self, rel_key):
        accumulo_rel_key = '{}_{}_{}'.format(rel_key.user, rel_key.program, rel_key.relation)

        if not self.connection:
            raise RuntimeError(
                "no schema for relation %s because no connection" % rel_key)

        props = self.connection.getTableProperties(accumulo_rel_key)
        # self.connection.client.getTableProperties(self.connection.self.login, accumulo_rel_key)
        print props

        # todo return list of tuples and types
        # props.howPartitioned
        return RepresentationProperties(
            hash_partitioned=frozenset( [] ))

