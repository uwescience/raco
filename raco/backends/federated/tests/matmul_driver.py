from raco.backends.spark.connection import SparkConnection
from raco.backends.spark.catalog import SparkCatalog
from raco.backends.spark.algebra import SparkAlgebra
from raco.backends.myria.connection import MyriaConnection
from raco.backends.myria.catalog import MyriaCatalog
from raco.backends.myria import MyriaLeftDeepTreeAlgebra
from raco.backends.federated.connection import FederatedConnection
from raco.backends.federated.catalog import FederatedCatalog
from raco.backends.federated import FederatedAlgebra
from raco.backends.federated.algebra import FederatedExec
from raco.compile import optimize

import raco.myrial.interpreter as interpreter
import raco.myrial.parser as myrialparser
from optparse import OptionParser

import raco.viz
import time
import os

def get_myria_connection():
    execution_url = os.environ.get('MYRIAX_REST_HOST', 'localhost')
    connection = MyriaConnection(hostname=execution_url, port=8753)
    return connection

def get_spark_connection():
    masterHostname = os.environ.get('sparkurl', 'localhost')
    if masterHostname == 'localhost':
        return SparkConnection(masterHostname)
    return SparkConnection("spark://{masterHostname}:7077".format(masterHostname=masterHostname))

# masterHostname = open("/root/spark-ec2/masters").read().strip()

program = """
matA = scan('{dataset1}');
matB = scan('{dataset2}');
A = [from matA emit col as col_a, row as row_a, value as val_a];
B = [from matB emit col as col_b, row as row_b, value as val_b];
outMat = [from A, B where A.col_a == B.row_b emit A.row_a as row, B.col_b as col, SUM(A.val_a*B.val_b) as value];
store(outMat, 'outMat.dat');
"""
program_fix = """
matA = scan('{dataset1}');
matB = scan('{dataset2}');
outMat = [from matA a, matB b emit a.row as arow, b.col as bcol];
store(outMat, 'outMat.dat');
"""
# dataset = 'hdfs://{masterhostname}:9000/data/{mat}'

myriaconn = get_myria_connection()
sparkconn = get_spark_connection()

myriacatalog = MyriaCatalog(myriaconn)
sparkcatalog = SparkCatalog(sparkconn)


catalog = FederatedCatalog([myriacatalog, sparkcatalog])

#matrices = ['random_N_10k_r_1.2.matrix.dat', 'random_N_50k_r_1.6.matrix.dat']
matrices = ['/Users/shrainik/Documents/Data/mat1','/Users/shrainik/Documents/Data/mat2']
#matrices = ['random_N_10k_r_1.2.matrix.dat', 'random_N_10k_r_1.2.matrix.dat', 'random_N_10k_r_1.3.matrix.dat', 'random_N_10k_r_1.4.matrix.dat', 'random_N_10k_r_1.5.matrix.dat', 'random_N_10k_r_1.6.matrix.dat', 'random_N_20k_r_1.2.matrix.dat', 'random_N_20k_r_1.3.matrix.dat', 'random_N_20k_r_1.4.matrix.dat', 'random_N_20k_r_1.5.matrix.dat', 'random_N_20k_r_1.6.matrix.dat', 'random_N_50k_r_1.2.matrix.dat', 'random_N_50k_r_1.3.matrix.dat', 'random_N_50k_r_1.4.matrix.dat', 'random_N_50k_r_1.5.matrix.dat', 'random_N_50k_r_1.6.matrix.dat' ]

print 'First experiment is just to startup spark... '
for mat in matrices:
    print 'Starting execution for: ', mat
    start = time.time()
    parser = myrialparser.Parser()
    processor = interpreter.StatementProcessor(catalog, True)
    myrial_code = program_fix.format(dataset1=mat, dataset2=mat)
    statement_list = parser.parse(myrial_code)
    processor.evaluate(statement_list)
    
    algebras = [MyriaLeftDeepTreeAlgebra(), SparkAlgebra()]
    falg = FederatedAlgebra(algebras, catalog)
    
    logical = processor.get_logical_plan()
    print 'Logical Plan: '
    print logical
    federated_plan = processor.get_physical_plan(target_alg=falg)
    
    dot_logical = raco.viz.operator_to_dot_object(logical)
    dot_federated = raco.viz.operator_to_dot_object(federated_plan)
    
    physical_plan_spark = optimize(federated_plan, SparkAlgebra())
    phys_dot = raco.viz.operator_to_dot_object(physical_plan_spark)
    print 'Physical Plan:'
    print physical_plan_spark
    sparkconn.execute_query(physical_plan_spark)
    end = time.time()
    total = end-start
    print 'Time Taken for just execute: ', total
