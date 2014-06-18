from abc import abstractmethod


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


class MyriaCatalog(object):

    @abstractmethod
    def get_num_servers(self):
        """ Return number of servers in myria deployment """

    @abstractmethod
    def num_tuples(self, rel_key):
        """ Return number of tuples of rel_key """


class FakeCatalog(MyriaCatalog):
    """ fake catlog, should only be used in test """
    def __init__(self, num_servers, child_sizes=None):
        self.num_servers = num_servers
        # default sizes
        self.cached = {}
        # overwrite default sizes if necessary
        if child_sizes:
            for child, size in child_sizes.items():
                self.cached["public:adhoc:{}".format(child)] = size

    def get_num_servers(self):
        return self.num_servers

    def num_tuples(self, rel_key):
        key = "{}:{}:{}".format(
            rel_key.user, rel_key.program, rel_key.relation)
        if key in self.cached:
            return self.cached[key]
        return 10000
