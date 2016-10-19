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
        schemestr = props['table.custom.scheme']
        sch = Scheme(eval(schemestr))
        return sch
        # sch2 = Scheme()
        #
        # def unmapTypes(s, c):
        #     name = c[0]
        #     if c[1] != "LONG_TYPE":
        #         _type = c[1]
        #     else:
        #         _type = "INT_TYPE"
        #     s.asdict[name] = (len(s.attributes), _type)
        #     s.attributes.append((name, _type))
        #
        # map(lambda c: unmapTypes(sch2, c), sch.attributes)
        # return sch2

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

        schemestr = props['table.custom.scheme']
        sch = Scheme(eval(schemestr))

        hp = map(lambda s: AttIndex(sch.getPosition(s)), eval(howPartitioned))


        return RepresentationProperties(
            hash_partitioned=frozenset(hp))
            # hash_partitioned=frozenset( eval(howPartitioned) ))

#conn.setTableProperty('public_adhoc_netflow','table.custom.scheme','[("TotBytes","INT_TYPE"),("StartTime","STRING_TYPE"),("SrcAddr","STRING_TYPE"),("DstAddr","STRING_TYPE"),("RATE","DOUBLE_TYPE"),("Dur","DOUBLE_TYPE"),("Dir","STRING_TYPE"),("Proto","STRING_TYPE"),("Sport","STRING_TYPE"),("Dport","STRING_TYPE"),("State","STRING_TYPE"),("sTos","INT_TYPE"),("dTos","INT_TYPE"),("TotPkts","INT_TYPE"),("SrcBytes","INT_TYPE"),("Label","STRING_TYPE")]')
#conn.setTableProperty('public_adhoc_netflow','table.custom.howPartitioned','["TotBytes", "StartTime"]')




