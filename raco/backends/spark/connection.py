import raco
from raco.compile import optimize
from raco.backends.spark.algebra import *
from raco.expression import *
import os, sys
import requests

__all__ = ['FederatedConnection']
# Path for spark source folder
# os.environ['SPARK_HOME']="your_spark_home_folder"

# Append pyspark to Python Path
sys.path.append(os.path.join(os.environ['SPARK_HOME'],"python"))


try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql.types import *
    from pyspark.sql import Row
    from pyspark.sql import SQLContext
    import raco.backends.spark.pyspark_csv as pycsv
    print ("Successfully imported Spark Modules")

except ImportError as e:
    raise

class SparkConnection(object):
    """A Spark connection wrapper"""

    def __init__(self, url, username=None, password=None):
        """
        Args:
            url: Spark URL
        """
        self.url = url
        self.masterhostname = url.split(':')[1][2:]
        self.context = SparkContext(self.url)
        # sparkcsv_python_file = os.path.join(os.path.dirname(__file__),"pyspark_csv.py")
        # self.sparkcsv_python_file = "hdfs://" + self.masterhostname + ":9000/pyspark_csv.py"
        # self.context.addPyFile(self.sparkcsv_python_file)
        self.sqlcontext = SQLContext(self.context)

    def get_df(self, df_name):
        # self.context.addPyFile(self.sparkcsv_python_file)
        # import pyspark_csv as pycsv
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

        # print "AFTER Spark RULES"
        # print physical_plan
        # print 'dot version after spark rules'
        # print raco.viz.operator_to_dot(physical_plan)
        #
        self.execute_rec(query)

    def condExprToSparkCond(self, leftdf, rightdf, plan, condition):
        # TODO: GENERALIZE TO OTHER CONDITIONS
        condition = condition.right
        left_cond = remove_unnamed_literals(plan, condition.left)
        right_cond = remove_unnamed_literals(plan, condition.right)
        if isinstance(condition, EQ):
            if left_cond in map(lambda p: p[0], plan.left.scheme().attributes):
                l_df = leftdf
            elif left_cond in map(lambda p: p[0], plan.right.scheme().attributes):
                l_df = rightdf
            if right_cond in map(lambda p: p[0], plan.left.scheme().attributes):
                r_df = leftdf
            elif right_cond in map(lambda p: p[0], plan.right.scheme().attributes):
                r_df = rightdf
            return [getattr(l_df, left_cond) == getattr(r_df, right_cond)]

    def execute_rec(self, plan):
        if isinstance(plan, SparkScan):
            if str(plan.relation_key).startswith('hdfs://'):
                return self.get_df(str(plan.relation_key))
            else:
                return self.get_df(str(plan.relation_key).split(':')[-1])
        if isinstance(plan, SparkScanTemp):
            df_temp = self.sqlcontext.sql("Select * from {}".format(plan.name))
            # if plan.name == 'prunedA':
            #     df_temp.show()
            if df_temp.count() == 1:
                if len(df_temp.dtypes) == 1:
                    if df_temp.dtypes[0][0] == plan.name + "_SINGLETON_RELATION_":
                        return self.sqlcontext.sql("select {} as {} from {}".format(plan.name + "_SINGLETON_RELATION_", plan.name, plan.name))
            return self.sqlcontext.sql("Select * from {}".format(plan.name))
        if isinstance(plan, SparkSelect):
            return self.execute_rec(plan.input).filter(remove_unnamed_literals(plan, plan.condition))
        if isinstance(plan, SparkProject):
            return self.execute_rec(plan.input).select([x.name for x in plan.columnlist])
        if isinstance(plan, SparkApply):
            if isinstance(plan.input, algebra.SingletonRelation):
                return plan
            temp_table_name = 'TempTable' + str(random.randint(1, 10000000))
            self.sqlcontext.registerDataFrameAsTable(self.execute_rec(plan.input), temp_table_name)
            # Todo: Fix expr to have proper aliases
            rename_str = ', '.join([remove_unnamed_literals(plan.input, expr) + ' as ' + str(col) for (col, expr) in plan.emitters])
            # print rename_str
            return self.sqlcontext.sql('select {} from {}'.format(rename_str, temp_table_name))
        if isinstance(plan, SparkGroupBy):
            agg_dict = {}
            for agg in plan.aggregate_list:
                if isinstance(agg, MIN):
                    agg_dict[str(agg.input)] = 'min'
                elif isinstance(agg, MAX):
                    agg_dict[str(agg.input)] = 'max'
                elif isinstance(agg, AVG):
                    agg_dict[str(agg.input)] = 'avg'
                elif isinstance(agg, COUNT):
                    agg_dict[str(agg.input)] = 'count'
                elif isinstance(agg, COUNTALL):
                    agg_dict['*'] = 'count'
                elif isinstance(agg, SUM):
                    agg_dict[str(agg.input)] = 'sum'
                else:
                    raise NotImplementedError("Aggregate not supported %s" % str(agg))
            if len(plan.grouping_list) == 0:
                # self.execute_rec(plan.input).agg(agg_dict).show()
                return self.execute_rec(plan.input).agg(agg_dict)
            else:
                gp_list = []
                for col in plan.grouping_list:
                    gp_list.append(str(col))
                # self.execute_rec(plan.input).groupBy(gp_list).agg(agg_dict).show()
                return self.execute_rec(plan.input).groupBy(gp_list).agg(agg_dict)
        if isinstance(plan, SparkJoin):
            # Todo: write separate cases for different types of join
            left = self.execute_rec(plan.left)
            right = self.execute_rec(plan.right)
            if remove_unnamed_literals(plan, plan.condition) == "(1 = 1)": # (I don't know why the condition is 1=1 cross product)
                return left.join(right)
            return left.join(right, self.condExprToSparkCond(left, right, plan, plan.condition))
        if isinstance(plan, SparkStore):
            # change with actual save later
            return self.execute_rec(plan.input).show(n=100)
            # return self.execute_rec(plan.input).rdd.saveAsTextFile(plan.relation_key.split(':')[-1])
        if isinstance(plan, SparkStoreTemp):
            # print plan.name
            if isinstance(plan.input, SparkApply):
                if isinstance(plan.input.input, algebra.SingletonRelation):
                    singletonRow = Row(plan.name + '_SINGLETON_RELATION_')
                    r = singletonRow(plan.input.emitters[0][1])
                    self.sqlcontext.createDataFrame(r).registerTempTable(plan.name + 'temp')
                    # print 'DSDADASD', self.sqlcontext.sql('Select value as {} from {}'.format(plan.name + '_SINGLETON_RELATION_', plan.name + 'temp')).show()
                    self.sqlcontext.sql('Select value as {} from {}'.format(plan.name + '_SINGLETON_RELATION_', plan.name + 'temp')).registerTempTable(plan.name)
                    return
            if isinstance(plan.input, SparkScanTemp):
                # Check if scantemp has just one column and it's name is the same as the relation name and that it has just one value
                # Assume that this is a singleton relation in this case
                df_temp = self.sqlcontext.sql("Select * from {}".format(plan.input.name))
                if df_temp.count() == 1:
                    if len(df_temp.dtypes) == 1:
                        if df_temp.dtypes[0][0] == plan.input.name + "_SINGLETON_RELATION_":
                            # print 'Assigning {} to {}'.format(plan.input.name, plan.name)
                            self.sqlcontext.sql("select {} as {} from {}".format(plan.input.name + "_SINGLETON_RELATION_", plan.name, plan.input.name)).registerTempTable(plan.name)
                            return
            # Check if scantemp has just one column and it's name _COLUMN0_ and that it has just one value
            # Assume that this is a singleton relation in this case
            df_temp = self.execute_rec(plan.input)
            if len(df_temp.dtypes) == 1:
                # print 'dtypes is 1'
                if df_temp.dtypes[0][0] == "_COLUMN0_":
                    # print 'column is _COLUMN0_'
                    if df_temp.count() == 1:
                        # print 'Assuming {} is a singleton relation...'.format(plan.name)
                        df_temp.registerTempTable(plan.name + 'temp')
                        self.sqlcontext.sql('select _COLUMN0_ as {} from {}'.format(plan.name + '_SINGLETON_RELATION_', plan.name + 'temp')).registerTempTable(plan.name)
                        # self.sqlcontext.sql('select * from {}'.format(plan.name)).show()
                        return
                    else:
                        pass
                        # print 'Count is ', df_temp.count()
            # Else, just register the dataframe created below this operator as a temp table
            # print "Else, just register the dataframe created below this operator as a temp table", plan.name
            self.sqlcontext.registerDataFrameAsTable(df_temp, plan.name)
            return
        if isinstance(plan, SparkSequence):
            for child in plan.children():
                self.execute_rec(child)
        if isinstance(plan, SparkDoWhile):
            cond = True
            num_children = len(plan.children())
            while(cond):
                for i in range(0,num_children-1):
                    self.execute_rec(plan.children()[i])
                cond = self.execute_rec(plan.children()[-1]).collect()[0]['_COLUMN0_']
                # print cond, type(cond)

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
