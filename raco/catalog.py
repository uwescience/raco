from abc import abstractmethod, ABCMeta
from ast import literal_eval
import os
import json

from raco.algebra import DEFAULT_CARDINALITY
from raco.representation import RepresentationProperties
from raco.relation_key import RelationKey
from raco.scheme import Scheme


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

    @abstractmethod
    def partitioning(self, rel_key):
        """ Return partitioning of rel_key """

    def representation_properties(self, rel_key):
        """
        Return interesting properties, like partitioning and sorted
        """
        # default is to return no information
        return RepresentationProperties()


# Some useful Catalog implementations
class FakeCatalog(Catalog):
    """ fake catalog, should only be used in test """

    def __init__(self, num_servers, child_sizes=None,
                 child_partitionings=None, child_functions=None):
        self.num_servers = num_servers
        # default sizes
        self.sizes = {}
        # default partitionings
        self.partitionings = {}
        # overwrite default sizes if necessary
        self.functions = {}

        if child_sizes:
            for child, size in child_sizes.items():
                self.sizes[RelationKey(child)] = size
        if child_partitionings:
            for child, part in child_partitionings.items():
                self.partitionings[RelationKey(child)] = frozenset(part)
        if child_functions:
            for child, typ in child_functions.items():
                self.functions[child] = funcObj

    def get_num_servers(self):
        return self.num_servers

    def num_tuples(self, rel_key):
        if rel_key in self.sizes:
            return self.sizes[rel_key]
        return DEFAULT_CARDINALITY

    def partitioning(self, rel_key):
        if rel_key in self.partitionings:
            return RepresentationProperties(
                hash_partitioned=self.partitionings[rel_key])
        return RepresentationProperties()

    def get_scheme(self, rel_key):
        raise NotImplementedError()

    def get_function(self, funcName):
        """Return UDF with name = funcName"""
        if funcName in self.functions:
            return self.functions[funcName]


class FromFileCatalog(Catalog):

    """ Catalog that is created from a python file.
    Format of file is a dictionary of schemas.

    {'relation1' : [('a', 'LONG_TYPE'), ('b', 'STRING_TYPE')],
     'relation2' : [('y', 'STRING_TYPE'), ('z', 'DATETIME_TYPE')]}

     Or there can be an optional cardinality for any relation
    {'relation1' : ([('a', 'LONG_TYPE'), ('b', 'STRING_TYPE')], 10),
     'relation2' : [('y', 'STRING_TYPE'), ('z', 'DATETIME_TYPE')]}

     Or it can be a single relation, using filename as basename
     [('a', 'LONG_TYPE'), ('b', 'STRING_TYPE')]

     see raco.types for allowed types
    """

    def __init__(self, cat, fname):
        self.catalog = {}

        def error():
            assert False, """Unexpected catalog file format. \
                    See raco.catalog.FromFileCatalog"""

        if isinstance(cat, dict):
            def parse(v):
                if isinstance(v, tuple):
                    return v
                elif isinstance(v, list):
                    return v, DEFAULT_CARDINALITY
                else:
                    error()

            self.catalog = dict([(k, parse(v)) for k, v in cat.iteritems()])
        elif isinstance(cat, list):
            name = os.path.splitext(os.path.basename(fname))[0]
            self.catalog = {
                'public:adhoc:{0}'.format(name): (
                    cat,
                    DEFAULT_CARDINALITY,
                )}
        else:
            error()

    def get_scheme(self, rel_key):
        return Scheme(self.__get_catalog_entry__(rel_key)[0])

    def get_keys(self):
        return self.catalog.keys()

    @classmethod
    def print_cat(cls, ffc1, ffc2):
        cpy = ffc1.catalog.copy()
        for k, v in ffc2.catalog.iteritems():
            cpy[k] = v
        print cpy

    @classmethod
    def load_from_file(cls, path):
        with open(path) as fh:
            return cls(literal_eval(fh.read()), path)

    @classmethod
    def scheme_write_to_file(cls, path, new_rel_key, new_rel_schema,
                             append=True):
        new_schema_entry = literal_eval(new_rel_schema)
        col_names = new_schema_entry['columnNames']
        col_types = new_schema_entry['columnTypes']
        columns = zip(col_names, col_types)

        if os.path.isfile(path):
            if append:
                schema_read = open(path, 'r')
                s = schema_read.read()
                schema_read.close()

                schema_write = open(path, 'w')
                current_dict = literal_eval(s)
                current_dict[new_rel_key] = columns
                json.dump(current_dict, schema_write)
                schema_write.write("\n")
                schema_write.close()
            else:
                raise IOError("file {0} exists".format(path))
        else:
            with open(path, 'w+') as fh:
                d = {}
                d[new_rel_key] = columns
                json.dump(d, fh)
                fh.write("\n")
            fh.close()

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

    def partitioning(self, rel_key):
        # TODO allow specifying an optional list of attributes
        return RepresentationProperties()
