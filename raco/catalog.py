from abc import abstractmethod, ABCMeta
from raco.relation_key import RelationKey


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
    def num_tuples(self, rel_key):
        """ Return number of tuples of rel_key """


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
        return 10000
