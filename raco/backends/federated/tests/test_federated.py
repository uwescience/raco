import unittest
from httmock import urlmatch, HTTMock
import json

from raco.backends.scidb.connection import SciDBConnection
from raco.backends.scidb.catalog import SciDBCatalog
from raco.backends.scidb.algebra import SciDBAFLAlgebra, SciDBScan, SciDBStore, SciDBConcat

from raco.backends.myria.connection import MyriaConnection
from raco.backends.myria.catalog import MyriaCatalog
from raco.backends.myria import MyriaLeftDeepTreeAlgebra

from raco.backends.federated.connection import FederatedConnection
from raco.backends.federated.catalog import FederatedCatalog
from raco.backends.federated import FederatedAlgebra
from raco.backends.federated.algebra import FederatedExec

import raco.myrial.interpreter as interpreter
import raco.myrial.parser as myrialparser
import raco.viz
from raco.backends.federated.movers.filesystem import SciDBToMyria

import os

program_simple = """
T1 = scan(abc);
T2 = [from T1 where value>40 emit value as X];
store(T2, JustX);
"""

program = """
const test_vector_id: 1;
const bins: 10;
vectors = scan(SciDB:Demo:Vectors);

-------------------------
-- Constants + Functions
-------------------------
const alpha: 1.0;

def log2(x): log(x) / log(2);
def mod2(x): x - int(x/2)*2;
def iif(expression, true_value, false_value):
    case when expression then true_value
         else false_value end;
def bucket(x, high, low): greater(least(int((bins-1) * (x - low) / iif(high != low, high - low, 1)),
                                bins - 1), 0);
def difference(current, previous, previous_time, time):
    iif(previous_time >= 0,
        (current - previous) * iif(previous_time < time, 1, -1),
        current);

symbols = empty(id:int, index:int, value:int);

------------------------------------------------------------------------------------
-- Harr Transform
------------------------------------------------------------------------------------
uda HarrTransformGroupBy(alpha, time, x) {
  [0.0 as coefficient, 0.0 as _sum, 0 as _count, -1 as _time];
  [difference(x, coefficient, _time, time), _sum + x, _count + 1, time];
  [coefficient, _sum / int(_count * alpha)];
};

iterations = [from vectors where id = test_vector_id emit 0 as i, int(ceil(log2(count(*)))) as total];
do
    groups = [from vectors emit
                     id,
                     int(floor(time/2)) as time,
                     HarrTransformGroupBy(alpha, time, value) as [coefficient, mean]];
    coefficients = [from groups emit id, coefficient];
    range = [from vectors emit max(value) - min(value) as high, min(value) - max(value) as low];
    histogram = [from coefficients, range
                 emit id,
                      bucket(coefficient, high, low) as index,
                      count(bucket(coefficient, high, low)) as value];
    symbols = symbols + [from histogram, iterations emit id, index + i*bins as index, value];
    vectors = [from groups emit id, time, mean as value];
    iterations = [from iterations emit $0 + 1, $1];
while [from iterations emit $0 < $1];

sink(symbols);

--========================================================================
-- Myria
--========================================================================

const test_vector_id1: 1;
def idf(w_ij, w_ijN, N): log(N / w_ijN) * w_ij;


------------------------------------------------------------------------------------
-- IDF
------------------------------------------------------------------------------------
ids = distinct([from symbols emit id]);
N = [from ids emit count(*) as N];
frequencies = [from symbols emit value, index, count(*) as frequency];

tfv = [from symbols, frequencies, N
       where symbols.value = frequencies.value
       emit id, index, idf(value, frequency, N) as value];

------------------------------------------------------------------------------------
-- Conditioning
------------------------------------------------------------------------------------
moments = [from tfv emit id,
                         avg(value) as mean,
                         -- Sample estimator
                         sqrt((stdev(value)*stdev(value)*count(value))/(count(value)-1)) as std];
conditioned_tfv = [from tfv, moments
                   where tfv.id = moments.id
                   emit id, index, value as v, mean, std, (value - mean) / std as value];
sum_squares = [from conditioned_tfv
               emit id, sum(pow(value, 2)) as sum_squares];

------------------------------------------------------------------------------------
-- k-NN
------------------------------------------------------------------------------------

test_vector = [from conditioned_tfv where id = test_vector_id1 emit *];

products = [from test_vector as x,
                 conditioned_tfv as y
                where x.index = y.index
                emit y.id as id, sum(x.value * y.value) as product];

correlations = [from products, sum_squares
                where products.id = sum_squares.id
                emit products.id as id, product / sum_squares as rho];

store(correlations, correlations);
-- sink(correlations);
"""

def skip(str):
    return not (str in os.environ
                and int(os.environ[str]) == 1)


def get_myria_connection():
    if skip('RACO_MYRIAX_TESTS'):
        # Use the local stub server
        # connection = MyriaConnection(hostname='localhost', port=12345)
        # rest_url = 'http://localhost:8753'
        # execution_url = 'http://localhost:8090'
        rest_url = 'http://ec2-52-1-38-182.compute-1.amazonaws.com:8753'
        execution_url = 'http://demo.myria.cs.washington.edu'
        connection = MyriaConnection(rest_url=rest_url,
                                     execution_url=execution_url)
    else:
        # Use the production server
        #rest_url = 'https://rest.myria.cs.washington.edu:1776'
        #execution_url = 'https://myria-web.appspot.com'
        rest_url = 'http://ec2-52-1-38-182.compute-1.amazonaws.com:8753'
        execution_url = 'http://demo.myria.cs.washington.edu'
        connection = MyriaConnection(rest_url=rest_url,
                                     execution_url=execution_url)

    return connection


def get_scidb_connection():
    if skip('RACO_SCIDB_TESTS'):
        # Use the local stub server
        connection = SciDBConnection('http://ec2-54-175-66-8.compute-1.amazonaws.com:8080')
        # connection = SciDBConnection('http://localhost:8751')

        # connection = SciDBConnection('http://localhost:9000')
    else:
        # Use the production server
        connection = SciDBConnection()

    return connection


def query(myriaconnection, scidbconnection):

    myriacatalog = MyriaCatalog(myriaconnection)
    scidbcatalog = SciDBCatalog(scidbconnection)

    catalog = FederatedCatalog([myriacatalog, scidbcatalog])

    parser = myrialparser.Parser()

    # TODO: StatementProcessor needs catalog to typecheck relation keys
    # but the Algebra/Rules need the catalog to during optimization
    # Do we really need it both places?
    processor = interpreter.StatementProcessor(catalog, True)

    statement_list = parser.parse(program_simple)
    #
    processor.evaluate(statement_list)
    #
    # # TODO: Should we just have every algebra take a catalog object as a parameter?
    algebras = [MyriaLeftDeepTreeAlgebra(), SciDBAFLAlgebra()]
    falg = FederatedAlgebra(algebras, catalog)
    #
    logical = processor.get_logical_plan()
    print "LOGICAL"
    print logical
    #
    pd = processor.get_physical_plan(target_alg=falg)
    # pd = processor.get_physical_plan(target_alg=SciDBAFLAlgebra())
    #
    print "PHYSICAL"
    print pd
    #
    # print raco.viz.operator_to_dot(pd)
    scidbconnection.execute_query(pd.args[0].plan)

    # fedconn = FederatedConnection([myriaconnection, scidbconnection], [SciDBToMyria()])

        # result = fedconn.execute_query(program)

    # return result
    # return logical

def empty_query():
    """Simple empty query"""
    return {'rawQuery': 'empty',
            'logicalRa': 'empty',
            'fragments': []}

def query_status(query, query_id=17, status='SUCCESS'):
    return {'url': 'http://localhost:12345/query/query-%d' % query_id,
            'queryId': query_id,
            'rawQuery': query['rawQuery'],
            'logicalRa': query['rawQuery'],
            'plan': query,
            'submitTime': '2014-02-26T15:19:54.505-08:00',
            'startTime': '2014-02-26T15:19:54.611-08:00',
            'finishTime': '2014-02-26T15:23:34.189-08:00',
            'elapsedNanos': 219577567891,
            'status': status}

@urlmatch(netloc=r'localhost:12345')
def local_mock(url, request):
    global query_counter
    query_request = empty_query()
    if url.path == '/query' and request.method == 'POST':
        # raise ValueError(type(request.body))
        body = query_status(json.loads(request.body), 17, 'ACCEPTED')
        headers = {'Location': 'http://localhost:12345/query/query-17'}
        query_counter = 2
        return {'status_code': 202, 'content': body, 'headers': headers}
    elif '/dataset' in url.path:
        relstr = url.path.split("/")[-3:]
        relkey = [key.split("-")[1] for key in relstr]
        if relkey[0] != "Brandon":
            return {'status_code': 404, 'content': ''}

        dataset_info = {
            'schema': {
                'columnNames': [u'id', u'time', u'value'],
                'columnTypes': ['LONG_TYPE', 'LONG_TYPE', 'DOUBLE_TYPE']
            },
            'numTuples': 500
        }
        return {'status_code': 200, 'content': dataset_info}
    elif url.path == '/query/query-17':
        if query_counter == 0:
            status = 'SUCCESS'
            status_code = 201
        else:
            status = 'ACCEPTED'
            status_code = 202
            query_counter -= 1
        body = query_status(query_request, 17, status)
        headers = {'Location': 'http://localhost:12345/query/query-17'}
        return {'status_code': status_code,
                'content': body,
                'headers': headers}
    elif url.path == '/query/validate':
        return request.body
    elif url.path == '/query' and request.method == 'GET':
        body = {'max': 17, 'min': 1,
                'results': [query_status(query_request, 17, 'ACCEPTED'),
                            query_status(query_request, 11, 'SUCCESS')]}
        return {'status_code': 200, 'content': body}

    return None




if __name__ == '__main__':
    with HTTMock(local_mock):
        myriaconn = get_myria_connection()
        scidbconn = get_scidb_connection()
        query(myriaconn, scidbconn)

