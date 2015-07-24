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

import raco.myrial.interpreter as interpreter
import raco.myrial.parser as myrialparser

from raco.backends.federated.movers.filesystem import SciDBToMyria

import os


def skip(str):
    return not (str in os.environ
                and int(os.environ[str]) == 1)


def get_myria_connection():
    if skip('RACO_MYRIAX_TESTS'):
        # Use the local stub server
        connection = MyriaConnection(hostname='localhost', port=12345)
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

    program = """
-------------------------
-- Constants + Functions
-------------------------
const bins: 10;

def iif(expression, true_value, false_value):
    case when expression then true_value
         else false_value end;
def bin(x, high, low): greater(least(int((bins-1) * (x - low) / iif(high != low, high - low, 1)),
                                bins - 1), 0);
def difference(current, previous, previous_time, time):
    iif(previous_time >= 0,
        (current - previous) * iif(previous_time < time, 1, -1),
        current);
uda HarrTransformGroupBy(time, x) {
  [0.0 as coefficient, 0.0 as _sum, 0 as _count, -1 as _time];
  [difference(x, coefficient, _time, time), _sum + x, _count + 1, time];
  [coefficient, _sum / int(_count)];
};

------------------------------------------------------------------------------------
-- Harr Transform
------------------------------------------------------------------------------------
vectors = scan(SciDB:Demo:Vectors);

groups = [from vectors emit
                 id,
                 int(floor(time/2)) as time,
                 HarrTransformGroupBy(time, value) as [coefficient, mean]];

histogram = [from groups
             emit id,
                  bin(coefficient, 1, 0) as index,
                  count(bin(coefficient, 1, 0)) as value];

-- *******************************
--- Added to test orchestrator
r = scan(Brandon:Demo:Vectors);
histogram = histogram + r;
-- *******************************

sink(histogram);
"""

    statement_list = parser.parse(program)

    processor.evaluate(statement_list)

    # TODO: Should we just have every algebra take a catalog object as a parameter?
    algebras = [MyriaLeftDeepTreeAlgebra(), SciDBAFLAlgebra()]
    falg = FederatedAlgebra(algebras, catalog)

    logical = processor.get_logical_plan()
    print "LOGICAL"
    print logical

    pd = processor.get_physical_plan(target_alg=falg)

    print "PHYSICAL"
    print pd

    fedconn = FederatedConnection([myriaconnection, scidbconnection], [SciDBToMyria()])

    result = fedconn.execute_query(pd)

    return result

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
