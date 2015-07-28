import scidbpy
from raco.compile import optimize
from raco.backends.scidb.algebra import SciDBAFLAlgebra, compile_to_afl, compile_to_afl_new

__all__ = ['FederatedConnection']

class SciDBConnection(object):
    """A SciDB connection wrapper"""

    def __init__(self, url, username=None, password=None):
        """
        Args:
            url: SciDB shim URL
        """
        self.connection = scidbpy.connect(url, username=username, password=password)

    def workers(self):
        """Return a dictionary of the workers"""
        keys = ('hostname', 'port', 'id', 'created', 'path')
        return [dict(izip(keys, values)) for values in self.connection.afl.list("'instances'").toarray()]

    def workers_alive(self):
        """Return a list of the workers that are alive"""
        return self.workers()

    def worker(self, worker_id):
        """Return information about the specified worker"""
        return next(worker for worker in self.workers() if worker['id'] == worker_id)

    def datasets(self):
        """Return a list of the datasets that exist"""
        return self.connection.list()

    def dataset(self, name):
        """Return information about the specified relation"""
        # TODO is name really a Myria relation triple?
        return self.connection.show(name)

    def download_dataset(self, name):
        """Download the data in the dataset as json"""
        # TODO is name really a Myria relation triple?
        return self.connection.wrap_array(name).todataframe().to_json()

    def submit_query(self, query):
        """Submit the query and return the status including the URL
        to be polled.

        Args:
            query: a physical plan as a Python object.
        """
        # TODO this blocks, and doesn't return a URL
        #return self.connection.query(query)
        raise NotImplemented

    def execute_query(self, query):
        """Submit the query and block until it finishes

        Args:
            query: a physical plan as a Python object.
        """
        # Assuming that the scidb part of the query plan will always be an store,
        # as we will do something with the result of scidb in myria,
        # hardcoding to optimize only the relation_key within store.
        # This relation_key is the plan for the entire scidb operation.

        # return self.connection.query(compile_to_afl(physical_plan))

        physical_plan = optimize(query, SciDBAFLAlgebra())
        # print "AFTER SCIDB RULES"
        # print physical_plan
        # # compile_to_afl_new(physical_plan)

        afl_string = compile_to_afl(physical_plan)
        result = ""
        # sci-db AFL parser expects one statement at a time
        for stmt in afl_string.split(";"):
            if len(stmt) <= 1:
                break
            result += str(self.connection.query(stmt))

        # FIXME: which do we want?
        return {
                 # myria-web
                'query_status': result,
                'query_url': 'TODO:scidb url',

                # myriaX response format
                'status': result,
                'url': 'TODO:scidb url'
        }

    def validate_query(self, query):
        """Submit the query to Myria for validation only.

        Args:
            query: a Myria physical plan as a Python object.
        """
        raise NotImplemented

    def get_query_status(self, query_id):
        """Get the status of a submitted query.

        Args:
            query_id: the id of a submitted query
        """
        raise NotImplemented

    def get_query_plan(self, query_id, subquery_id):
        """Get the saved execution plan for a submitted query.

        Args:
            query_id: the id of a submitted query
            subquery_id: the subquery id within the specified query
        """
        raise NotImplemented

    def get_sent_logs(self, query_id, fragment_id=None):
        """Get the logs for where data was sent.

        Args:
            query_id: the id of a submitted query
            fragment_id: the id of a fragment
        """
        raise NotImplemented

    def get_profiling_log(self, query_id, fragment_id=None):
        """Get the logs for operators.

        Args:
            query_id: the id of a submitted query
            fragment_id: the id of a fragment
        """
        raise NotImplemented

    def get_profiling_log_roots(self, query_id, fragment_id):
        """Get the logs for root operators.

        Args:
            query_id: the id of a submitted query
            fragment_id: the id of a fragment
        """
        raise NotImplemented

    def queries(self, limit=None, max_id=None, min_id=None, q=None):
        """Get count and information about all submitted queries.

        Args:
            limit: the maximum number of query status results to return.
            max_id: the maximum query ID to return.
            min_id: the minimum query ID to return. Ignored if max_id is
                    present.
            q: a text search for the raw query string.
        """
        raise NotImplemented

    def upload_file(self, relation_key, schema, data, overwrite=None,
                    delimiter=None, binary=None, is_little_endian=None):
        """Upload a file in a streaming manner to Myria.

        Args:
            relation_key: relation to be created.
            schema: schema of the relation.
            data: the bytes to be uploaded.
            overwrite: optional boolean indicating that an existing relation
                should be overwritten. Myria default is False.
            delimiter: optional character which delimits a CSV file. Only valid
                if binary is False. Myria default is ','.
            binary: optional boolean indicating that the data is encoded as
                a packed binary. Myria default is False.
            is_little_endian: optional boolean indicating that the binary data
                is in little-Endian. Myria default is False.
        """
        raise NotImplemented
