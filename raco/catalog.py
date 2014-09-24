from abc import abstractmethod, ABCMeta
from raco.algebra import DEFAULT_CARDINALITY
from raco.relation_key import RelationKey
from raco.scheme import Scheme
from ast import literal_eval


class Relation(object):
    def __init__(self, name, sch):
        self.name = name
        self._scheme = sch

    def __eq__(self, other):
        return self.name == other.name
        # and self.scheme == other.scheme

    def scheme(self):
        return self._scheme


class FileRelation(Relation):
    pass


class ASCIIFile(FileRelation):
    pass


class Catalog(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_num_servers(self):
        """ Return number of servers in myria deployment """

    @abstractmethod
    def get_scheme(self, rel_key):
        """ Return scheme of tuples of rel_key """

    @abstractmethod
    def num_tuples(self, rel_key):
        """ Return number of tuples of rel_key """


# Some useful Catalog implementations

class FakeCatalog(Catalog):
    """ fake catalog, should only be used in test """
    def __init__(self, num_servers, child_sizes=None):
        self.num_servers = num_servers
        # default sizes
        self.cached = {}
        # overwrite default sizes if necessary
        if child_sizes:
            for child, size in child_sizes.items():
                self.cached[RelationKey(child)] = size

    def get_num_servers(self):
        return self.num_servers

    def num_tuples(self, rel_key):
        if rel_key in self.cached:
            return self.cached[rel_key]
        return DEFAULT_CARDINALITY

    def get_scheme(self, rel_key):
        raise NotImplementedError()


class FromFileCatalog(Catalog):
    """ Catalog that is created from a python file.
    Format of file is a dictionary of schemas.

    {'relation1' : [('a', 'LONG_TYPE'), ('b', 'STRING_TYPE')],
     'relation2' : [('y', 'STRING_TYPE'), ('z', 'DATETIME_TYPE')]}

     Or there can be an optional cardinality for any relation
    {'relation1' : ([('a', 'LONG_TYPE'), ('b', 'STRING_TYPE')], 10),
     'relation2' : [('y', 'STRING_TYPE'), ('z', 'DATETIME_TYPE')]}

     see raco.types for allowed types
    """

    def __init__(self, cat):
        self.catalog = {}

        def parse(v):
            if isinstance(v, tuple):
                return v
            elif isinstance(v, list):
                return v, DEFAULT_CARDINALITY
            else:
                assert False, """Unexpected catalog file format. \
                See raco.catalog.FromFileCatalog"""

        self.catalog = dict([(k, parse(v)) for k, v in cat.iteritems()])

    def get_scheme(self, rel_key):
        return Scheme(self.__get_catalog_entry__(rel_key)[0])

    @classmethod
    def load_from_file(cls, path):
        with open(path) as fh:
            return cls(literal_eval(fh.read()))

    def get_num_servers(self):
        return 1

    def __get_catalog_entry__(self, rel_key):
        e = self.catalog.get(str(rel_key))
        if e is None:
            raise Exception(
                "relation {r} not found in catalog".format(r=rel_key))

        return e

    def num_tuples(self, rel_key):
        return self.__get_catalog_entry__(rel_key)[1]
