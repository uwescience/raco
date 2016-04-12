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
matA = scan('hdfs://{masterhostname}:9000/data/random_N_50k_r_1.2.matrix.dat');
matB = scan('hdfs://{masterhostname}:9000/data/random_N_50k_r_1.2.matrix.dat');
matC = scan('hdfs://{masterhostname}:9000/data/random_N_50k_r_1.2.matrix.dat');
A = [from matA emit col as col_a, row as row_a, value as val_a];
B = [from matB emit col as col_b, row as row_b, value as val_b];
C = [from matC emit col as col_c, row as row_c, value as val_c];
outMat = [from A, B, C where A.col_a = B.row_b and B.col_b = C.row_c emit A.row_a as row, C.col_c as col, SUM(A.val_a*B.val_b*C.val_c) as value];
store(outMat, '/data/outMat.dat');
""".format(masterhostname=masterHostname)

myriaconn = get_myria_connection()
sparkconn = get_spark_connection()

myriacatalog = MyriaCatalog(myriaconn)
sparkcatalog = SparkCatalog(sparkconn)

myrial_code = program_mcl

catalog = FederatedCatalog([myriacatalog, sparkcatalog])
parser = myrialparser.Parser()
processor = interpreter.StatementProcessor(catalog, True)
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

start = time.time()
sparkconn.execute_query(physical_plan_spark)
end = time.time()
print 'Time Taken for just execute: ', (end-start)
