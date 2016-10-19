from raco.catalog import Catalog
from raco.representation import RepresentationProperties
from raco.expression import UnnamedAttributeRef as AttIndex
from raco.catalog import DEFAULT_CARDINALITY
from raco.scheme import Scheme


class AccumuloCatalog(Catalog):

    def __init__(self, connection):
        self.connection = connection

    def get_scheme(self, rel_key):
        #print rel_key
        accumulo_rel_key = '{}_{}_{}'.format(rel_key.user, rel_key.program, rel_key.relation)

        if not self.connection:
            raise RuntimeError(
                "no schema for relation %s because no connection" % rel_key)

        props = self.connection.getTableProperties(accumulo_rel_key)
        scheme = props['table.custom.scheme']
        return Scheme(eval(scheme))

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
        howPartitioned = props['table.custom.howPartitioned']

        return RepresentationProperties(
            hash_partitioned=frozenset( eval(howPartitioned) ))

#conn.setTableProperty('public_adhoc_netflow','table.custom.scheme','[("TotBytes","LONG_TYPE"),("StartTime","STRING_TYPE"),("SrcAddr","STRING_TYPE"),("DstAddr","STRING_TYPE"),("RATE","DOUBLE_TYPE"),("Dur","DOUBLE_TYPE"),("Dir","STRING_TYPE"),("Proto","STRING_TYPE"),("Sport","STRING_TYPE"),("Dport","STRING_TYPE"),("State","STRING_TYPE"),("sTos","LONG_TYPE"),("dTos","LONG_TYPE"),("TotPkts","LONG_TYPE"),("SrcBytes","LONG_TYPE"),("Label","STRING_TYPE")]')
#conn.setTableProperty('public_adhoc_netflow','table.custom.howPartitioned','["TotBytes", "StartTime"]')




