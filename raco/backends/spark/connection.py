import raco
from raco.compile import optimize
from raco.backends.spark.algebra import *
import os, sys
import requests

__all__ = ['FederatedConnection']
# Path for spark source folder
# os.environ['SPARK_HOME']="your_spark_home_folder"

# Append pyspark to Python Path
sys.path.append("/Users/shrainik/Downloads/spark-1.6.0-bin-hadoop2.6/python")
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql.types import *
    from pyspark.sql import SQLContext
    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Cannot import Spark Modules", e)

class SparkConnection(object):
    """A Spark connection wrapper"""

    def __init__(self, url, username=None, password=None):
        """
        Args:
            url: Spark URL
        """
        self.url = url
        self.context = SparkContext('local')
        self.context.addPyFile('/Users/shrainik/Dropbox/raco/raco/backends/spark/pyspark_csv.py')
        import pyspark_csv as pycsv
        self.sqlcontext = SQLContext(self.context)

    def get_df(self, df_name):
        import pyspark_csv as pycsv
        return pycsv.csvToDataFrame(self.sqlcontext, self.context.textFile(df_name))

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

        physical_plan = optimize(query, SparkAlgebra())
        print "AFTER Spark RULES"
        print physical_plan
        print 'dot version after spark rules'
        print raco.viz.operator_to_dot(physical_plan)


        self.execute_rec(physical_plan)

    def execute_rec(self, plan):
        if isinstance(plan, SparkScan):
            return self.get_df(str(plan.relation_key).split(':')[-1])
        if isinstance(plan, SparkSelect):
            return self.execute_rec(plan.input).filter(remove_unnamed_literals(plan.scheme(), plan.condition))
        if isinstance(plan, SparkProject):
            print ", ".join([x.name for x in plan.columnlist])
            return self.execute_rec(plan.input).select(*[x.name for x in plan.columnlist])
        if isinstance(plan, SparkStore):
            # change with actual save later
            return self.execute_rec(plan.input).show()
            # return self.execute_rec(plan.input).rdd.saveAsTextFile(plan.condition.split(':')[-1])

    def remove_unnamed_literals(scheme, expression):
        ex = str(expression)
        for i in range(len(scheme)):
                unnamed_literal = "$" + str(i)
                ex = ex.replace(unnamed_literal, scheme.getName(i))
        return ex

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
