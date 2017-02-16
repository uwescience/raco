from raco.catalog import Catalog
import raco.scheme as scheme
from raco.representation import RepresentationProperties
from raco.expression import UnnamedAttributeRef as AttIndex
from raco.catalog import DEFAULT_CARDINALITY
from .errors import MyriaError


class MyriaCatalog(Catalog):

    def __init__(self, connection):
        self.connection = connection

    def get_scheme(self, rel_key):
        relation_args = {
            'userName': rel_key.user,
            'programName': rel_key.program,
            'relationName': rel_key.relation
        }
        if not self.connection:
            raise RuntimeError(
                "no schema for relation %s because no connection" % rel_key)
        try:
            dataset_info = self.connection.dataset(relation_args)
        except MyriaError:
            raise ValueError('No relation {} in the catalog'.format(rel_key))
        schema = dataset_info['schema']
        return scheme.Scheme(zip(schema['columnNames'], schema['columnTypes']))

    def get_num_servers(self):
        if not self.connection:
            raise RuntimeError("no connection.")
        return len(self.connection.workers_alive())

    def get_function(self, name):
        """ Get user defined function metadata """
        if not self.connection:
            raise RuntimeError("no connection.")

        try:
            function_info = self.connection.get_function(name)
        except MyriaError:
            raise ValueError("Function does not exist.")

        return function_info

    def num_tuples(self, rel_key):
        relation_args = {
            'userName': rel_key.user,
            'programName': rel_key.program,
            'relationName': rel_key.relation
        }
        if not self.connection:
            raise RuntimeError(
                "no cardinality of %s because no connection" % rel_key)
        try:
            dataset_info = self.connection.dataset(relation_args)
        except MyriaError:
            raise ValueError(rel_key)
        num_tuples = dataset_info['numTuples']
        assert isinstance(num_tuples, (int, long)), type(num_tuples)
        # that's a work round. numTuples is -1 if the dataset is old
        if num_tuples != -1:
            assert num_tuples >= 0
            return num_tuples
        return DEFAULT_CARDINALITY

    def partitioning(self, rel_key):
        relation_args = {
            'userName': rel_key.user,
            'programName': rel_key.program,
            'relationName': rel_key.relation
        }
        if not self.connection:
            raise RuntimeError(
                "no schema for relation %s because no connection" % rel_key)
        try:
            dataset_info = self.connection.dataset(relation_args)
        except MyriaError:
            raise ValueError('No relation {} in the catalog'.format(rel_key))
        distribute_function = dataset_info['howDistributed']['df']
        # TODO: can we do anything useful with other distribute functions?
        if distribute_function:
            if distribute_function['type'] == "Broadcast":
                return RepresentationProperties(broadcasted=True)
            if distribute_function['type'] == "Identity":
                index = distribute_function['index']
                return RepresentationProperties(
                    hash_partitioned=frozenset(AttIndex(index)))
            elif distribute_function['type'] == "Hash":
                indexes = distribute_function['indexes']
                return RepresentationProperties(
                    hash_partitioned=frozenset(AttIndex(i) for i in indexes))
        return RepresentationProperties()
