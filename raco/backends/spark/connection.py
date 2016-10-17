import raco
from raco.compile import optimize
from raco.backends.spark.algebra import *
from raco.expression import *
import time
__all__ = ['FederatedConnection']
# Path for spark source folder
# os.environ['SPARK_HOME']="your_spark_home_folder"

# Append pyspark to Python Path
#sys.path.append(os.path.join(os.environ['SPARK_HOME'],"python"))

import os
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql import SQLContext
### Adding helper methods csv taken from pyspark_csv.py
#TODO: Fix later to figure out how to add a file

class SparkConnection(object):
    """A Spark connection wrapper"""

    def __init__(self, url, username=None, password=None):
        """
        Args:
            url: Spark URL
        """
        self.url = url
        if url == 'localhost':
            self.masterhostname = url
            self.url = 'local[4]'
        else:
            self.masterhostname = url.split(':')[1][2:]
        self.context = SparkContext(self.url)
        self.context.setLogLevel("WARN")
        self.sqlcontext = SQLContext(self.context)
        self.singletons = []

    def get_df(self, df_name):
        print self.masterhostname, df_name # temporary to check masterhostname assignment
        if self.masterhostname=='localhost':
            rel_location = df_name
        else:
            rel_location="hdfs://{master}:9000/{rel}".format(master=self.masterhostname, rel=df_name)
        df = self.sqlcontext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(rel_location)
        df.cache()
        return df

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
        return self.execute_rec(query)

    def remove_unnamed_literals(scheme, expression):
        ex = str(expression)
        for i in range(len(scheme)):
                unnamed_literal = "$" + str(i)
                ex = ex.replace(unnamed_literal, scheme.getName(i))
        return ex

    def condExprToSparkCond(self, leftdf, rightdf, plan, condition):
        # TODO: GENERALIZE TO OTHER CONDITIONS
        if isinstance(condition, EQ):
            left_cond = remove_unnamed_literals(plan, condition.left)
            right_cond = remove_unnamed_literals(plan, condition.right)
            if left_cond == '1' and right_cond == '1':
                return [True]
            if left_cond in map(lambda p: p[0], leftdf.dtypes):
                l_df = leftdf
            elif left_cond in map(lambda p: p[0], rightdf.dtypes):
                l_df = rightdf
            if right_cond in map(lambda p: p[0], leftdf.dtypes):
                r_df = leftdf
            elif right_cond in map(lambda p: p[0], rightdf.dtypes):
                r_df = rightdf
            return [getattr(l_df, left_cond) == getattr(r_df, right_cond)]
        elif isinstance(condition, NEQ):
            left_cond = remove_unnamed_literals(plan, condition.left)
            right_cond = remove_unnamed_literals(plan, condition.right)
            if left_cond in map(lambda p: p[0], leftdf.dtypes):
                l_df = leftdf
            elif left_cond in map(lambda p: p[0], rightdf.dtypes):
                l_df = rightdf
            if right_cond in map(lambda p: p[0], leftdf.dtypes):
                r_df = leftdf
            elif right_cond in map(lambda p: p[0], rightdf.dtypes):
                r_df = rightdf
            return [getattr(l_df, left_cond) != getattr(r_df, right_cond)]
        elif isinstance(condition, AND):
            l = self.condExprToSparkCond(leftdf, rightdf, plan, condition.left)
            r = self.condExprToSparkCond(leftdf, rightdf, plan, condition.right)
            return l + r

    def matchOperatorAndDataFrameScheme(self, plan, leftdf, rightdf):
        n_leftdf = leftdf
        n_rightdf = rightdf
        plan_scheme = plan.scheme()
        # Change the right df, ie, rename columns, if necessary, nothing required to be done for left df.
        assert len(n_leftdf.columns) == len(plan.left.scheme().get_names())
        assert len(n_rightdf.columns) == len(plan.right.scheme().get_names())
        cols = plan_scheme.get_names()
        if len(n_rightdf.columns) > 0:
            c = 0
            proj_list = []
            for i in range(len(n_leftdf.columns), len(cols)):
                proj_list.append('{col} as {n_col}'.format(col=n_rightdf.columns[c], n_col=cols[i]))
                c+=1
            n_rightdf = rightdf.selectExpr(*(proj_list))
        return (n_leftdf, n_rightdf)

    def execute_rec(self, plan):
        if isinstance(plan, SparkScan):
            return self.get_df(plan.relation_key.relation)
        if isinstance(plan, SparkScanTemp):
            df_temp = self.sqlcontext.sql("Select * from {}".format(plan.name))
            # if plan.name == 'prunedA':
            #     df_temp.show()
            print 'SparkScanTemp:', plan.name
            #df_temp.show(n=10)
            if plan.name in self.singletons:
                return self.sqlcontext.sql("select {} as {} from {}".format(plan.name + "_SINGLETON_RELATION_", plan.name, plan.name))
            #if df_temp.count() == 1:
            #    if len(df_temp.dtypes) == 1:
            #        if df_temp.dtypes[0][0] == plan.name + "_SINGLETON_RELATION_":
            #            return self.sqlcontext.sql("select {} as {} from {}".format(plan.name + "_SINGLETON_RELATION_", plan.name, plan.name))
            return df_temp
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
            return self.sqlcontext.sql('select {} from {}'.format(rename_str, temp_table_name))
        if isinstance(plan, SparkGroupBy):
            agg_dict = {}
            for agg in plan.aggregate_list:
                if isinstance(agg, MIN):
                    agg_dict[remove_unnamed_literals(plan.input, agg.input)] = 'min'
                elif isinstance(agg, MAX):
                    agg_dict[remove_unnamed_literals(plan.input, agg.input)] = 'max'
                elif isinstance(agg, AVG):
                    agg_dict[remove_unnamed_literals(plan.input, agg.input)] = 'avg'
                elif isinstance(agg, COUNT):
                    agg_dict[remove_unnamed_literals(plan.input, agg.input)] = 'count'
                elif isinstance(agg, COUNTALL):
                    agg_dict['*'] = 'count'
                elif isinstance(agg, SUM):
                    agg_dict[remove_unnamed_literals(plan.input, agg.input)] = 'sum'
                else:
                    raise NotImplementedError("Aggregate not supported %s" % str(agg))
            if len(plan.grouping_list) == 0:
                # self.execute_rec(plan.input).agg(agg_dict).show()
                return self.execute_rec(plan.input).agg(agg_dict)
            else:
                gp_list = []
                for col in plan.grouping_list:
                    gp_list.append(remove_unnamed_literals(plan.input, col))
                # self.execute_rec(plan.input).groupBy(gp_list).agg(agg_dict).show()
                return self.execute_rec(plan.input).groupBy(gp_list).agg(agg_dict)
        if isinstance(plan, SparkJoin):
            # Todo: write separate cases for different types of join
            left = self.execute_rec(plan.left)
            right = self.execute_rec(plan.right)
            left, right = self.matchOperatorAndDataFrameScheme(plan, left, right)
            if remove_unnamed_literals(plan, plan.condition) == "(1 = 1)": # (I don't know why the condition is 1=1 cross product)
                return left.join(right)
            return left.join(right, self.condExprToSparkCond(left, right, plan, plan.condition)[1:])
        if isinstance(plan, SparkStore):
            result = self.execute_rec(plan.input)
            count =  result.count()
            result.show(n=1000)
            # TEMP FIX to support overwrite
            if self.masterhostname == 'localhost':
                return (count, str(plan.relation_key).split(':')[-1])
            os.system('~/ephemeral-hdfs/bin/hadoop fs -rmr /user/root/'+str(plan.relation_key).split(':')[-1]);
            result.rdd.saveAsTextFile(str(plan.relation_key).split(':')[-1])
            return (count, str(plan.relation_key).split(':')[-1])
        if isinstance(plan, SparkStoreTemp):
            print 'SparkStoreTemp', plan.name
            if isinstance(plan.input, SparkApply):
                if isinstance(plan.input.input, algebra.SingletonRelation):
                    self.singletons.append(plan.name)
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
                if len(df_temp.dtypes) == 1:
                    if df_temp.dtypes[0][0] == plan.input.name + "_SINGLETON_RELATION_":
                        # print 'Assigning {} to {}'.format(plan.input.name, plan.name)
                        self.sqlcontext.sql("select {} as {} from {}".format(plan.input.name + "_SINGLETON_RELATION_", plan.name, plan.input.name)).registerTempTable(plan.name)
                        return
                #if df_temp.count() == 1:
                #    if len(df_temp.dtypes) == 1:
                #        if df_temp.dtypes[0][0] == plan.input.name + "_SINGLETON_RELATION_":
                #            # print 'Assigning {} to {}'.format(plan.input.name, plan.name)
                #            self.sqlcontext.sql("select {} as {} from {}".format(plan.input.name + "_SINGLETON_RELATION_", plan.name, plan.input.name)).registerTempTable(plan.name)
                #            return
            # Check if scantemp has just one column and it's name _COLUMN0_ and that it has just one value
            # Assume that this is a singleton relation in this case
            df_temp = self.execute_rec(plan.input)
            #print 'Count of df_temp', df_temp.count()
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
                count, name = self.execute_rec(child)
            return (count, name)
        if isinstance(plan, SparkDoWhile):
            cond = True
            num_children = len(plan.children())
            num_iterations = 4
            itercount = 1
            count = None
            name = None
            while(cond and itercount<num_iterations):
                for i in range(0,num_children-1):
                    print 'child: ', i,':      ', plan.children()[i]
                    count, name = self.execute_rec(plan.children()[i])
                print 'Evaluating condition: ', plan.children()[-1]
                cond = self.execute_rec(plan.children()[-1]).first()['_COLUMN0_']
                #cond = self.execute_rec(plan.children()[-1]).collect()[0]['_COLUMN0_']
                num_iterations += 1
                # print cond, type(cond)
            return (count, name)

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
