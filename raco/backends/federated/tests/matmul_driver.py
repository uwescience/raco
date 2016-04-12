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
    rest_url = 'https://rest.myria.cs.washington.edu:1776'
    execution_url = 'http://demo.myria.cs.washington.edu'
    connection = MyriaConnection(rest_url=rest_url,
                                 execution_url=execution_url)
    return connection

def get_spark_connection():
    master = open("/root/spark-ec2/cluster-url").read().strip()
    connection = SparkConnection(master)
    return connection

masterHostname = open("/root/spark-ec2/masters").read().strip()

program_mcl = """
matA = scan('hdfs://{masterhostname}:9000/data/{mat}');
matB = scan('hdfs://{masterhostname}:9000/data/{mat}');
A = [from matA emit col as col_a, row as row_a, value as val_a];
B = [from matB emit col as col_b, row as row_b, value as val_b];
outMat = [from A, B where A.col_a == B.row_b emit A.row_a as row, B.col_b as col, SUM(A.val_a*B.val_b) as value];
store(outMat, '/data/outMat.dat');
"""

myriaconn = get_myria_connection()
sparkconn = get_spark_connection()

myriacatalog = MyriaCatalog(myriaconn)
sparkcatalog = SparkCatalog(sparkconn)


catalog = FederatedCatalog([myriacatalog, sparkcatalog])

matrices = ['random_N_10k_r_1.2.matrix.dat', 'random_N_10k_r_1.2.matrix.dat', 'random_N_10k_r_1.3.matrix.dat', 'random_N_10k_r_1.4.matrix.dat', 'random_N_10k_r_1.5.matrix.dat', 'random_N_10k_r_1.6.matrix.dat', 'random_N_20k_r_1.2.matrix.dat', 'random_N_20k_r_1.3.matrix.dat', 'random_N_20k_r_1.4.matrix.dat', 'random_N_20k_r_1.5.matrix.dat', 'random_N_20k_r_1.6.matrix.dat', 'random_N_50k_r_1.2.matrix.dat', 'random_N_50k_r_1.3.matrix.dat', 'random_N_50k_r_1.4.matrix.dat', 'random_N_50k_r_1.5.matrix.dat', 'random_N_50k_r_1.6.matrix.dat' ]

print 'First experiment is just to startup spark... '
for mat in matrices:
    print 'Starting execution for: ', mat
    start = time.time()
    parser = myrialparser.Parser()
    processor = interpreter.StatementProcessor(catalog, True)
    myrial_code = program_mcl.format(masterhostname=masterHostname, mat=mat)
    statement_list = parser.parse(myrial_code)
    processor.evaluate(statement_list)
    
    algebras = [MyriaLeftDeepTreeAlgebra(), SparkAlgebra()]
    falg = FederatedAlgebra(algebras, catalog)
    
    logical = processor.get_logical_plan()
    federated_plan = processor.get_physical_plan(target_alg=falg)
    
    dot_logical = raco.viz.operator_to_dot_object(logical)
    dot_federated = raco.viz.operator_to_dot_object(federated_plan)
    
    physical_plan_spark = optimize(federated_plan, SparkAlgebra())
    phys_dot = raco.viz.operator_to_dot_object(physical_plan_spark)
    sparkconn.execute_query(physical_plan_spark)
    end = time.time()
    total = end-start
    print 'Time Taken for just execute: ', total
