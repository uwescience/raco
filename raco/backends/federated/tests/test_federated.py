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
        rest_url = 'https://rest.myria.cs.washington.edu:1776'
        execution_url = 'https://myria-web.appspot.com'
        #rest_url = 'https://demo.myria.cs.washington.edu:1776'
        #execution_url = 'https://myria-web.appspot.com'
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
AA = scan(Brandon:Demo:Vectors);
A = select * from AA where value != 0;
B = scan(SciDB:Demo:Waveform);
mult = select A.i, B.j, sum(A.value*B.value) from A, B where A.j = B.i;
store(mult,Brandon:Demo:Mult);
"""

    statement_list = parser.parse(program)

    processor.evaluate(statement_list)

    # TODO: Should we just have every algebra take a catalog object as a parameter?
    algebras = [MyriaLeftDeepTreeAlgebra(), SciDBAFLAlgebra()]
    falg = FederatedAlgebra(algebras, catalog)

    logical = processor.get_logical_plan()
    print logical
    print "PHYSICAL"

    pd = processor.get_physical_plan(target_alg=falg)

    print pd

    fedconn = FederatedConnection([myriaconnection, scidbconnection], [SciDBToMyria()])

    result = None # fedconn.execute_query(pd)

    return result

@urlmatch(netloc=r'localhost:12345')
def local_mock(url, request):
    global query_counter
    global query_request
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
                'columnNames': [u'i', u'j', u'value'],
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
        print query(myriaconn, scidbconn)
