from httmock import urlmatch, HTTMock
import unittest
import json

from raco.backends.myria.connection import MyriaConnection
from raco.backends.myria.catalog import MyriaCatalog
from raco.relation_key import RelationKey
from raco.representation import RepresentationProperties

import raco.myrial.interpreter as interpreter
import raco.myrial.parser as myrialparser
from raco.backends.myria import MyriaLeftDeepTreeAlgebra
from raco.backends.myria.connection import FunctionTypes
import os


def is_skipping():
    return not ('RACO_MYRIAX_TESTS' in os.environ
                and int(os.environ['RACO_MYRIAX_TESTS']) == 1)


def get_connection():
    if is_skipping():
        # Use the local stub server
        connection = MyriaConnection(hostname='localhost', port=12345)
    else:
        # Use the production server
        rest_url = 'https://rest.myria.cs.washington.edu:1776'
        execution_url = 'https://myria-web.appspot.com'
        connection = MyriaConnection(rest_url=rest_url,
                                     execution_url=execution_url)

    return connection


def rel_info(connection):
    cat = MyriaCatalog(connection)
    key = RelationKey('Brandon', 'Demo', 'MoreBooks')
    return cat.num_tuples(key), cat.partitioning(key)


def query(connection):
    # Get the physical plan for a test query
    catalog = MyriaCatalog(connection)
    servers = catalog.get_num_servers()

    parser = myrialparser.Parser()
    processor = interpreter.StatementProcessor(catalog, True)

    program = """
        books = scan(Brandon:Demo:MoreBooks);
        longerBooks = [from books where pages > 300 emit name];
        store(longerBooks, Brandon:Demo:LongerBooks);
        """

    statement_list = parser.parse(program)

    processor.evaluate(statement_list)

    pd = processor.get_physical_plan(target_alg=MyriaLeftDeepTreeAlgebra())

    json = processor.get_json_from_physical_plan(pd)

    return {'rawQuery': program,
            'logicalRa': 'empty',
            'fragments': json}


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


query_counter = 0
# A real server would have the query info;
# here we cheat with a global
query_request = None


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
        dataset_info = {
            'schema': {
                'columnNames': [u'name', u'pages'],
                'columnTypes': ['STRING_TYPE', 'LONG_TYPE']
            },
            'howDistributed': {
                'df': None,
                'workers': None
            },
            'numTuples': 50
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
        return {'status_code': 200, 'content': request.body or ""}
    elif url.path == '/workers/alive':
        return {'status_code': 200, 'content': json.dumps([4])}
    elif url.path == '/query' and request.method == 'GET':
        body = {'max': 17, 'min': 1,
                'results': [query_status(query_request, 17, 'ACCEPTED'),
                            query_status(query_request, 11, 'SUCCESS')]}
        return {'status_code': 200, 'content': body}
    elif url.path == '/logs/sent' and request.method == 'GET':
        # lazy test
        return {'status_code': 200, 'content': request.body or ""}
    elif '/subquery' in url.path:
        # lazy test
        return {'status_code': 200, 'content': {'plan': 'fakeplan'}}
    elif url.path == '/logs/profilingroots' and request.method == 'GET':
        # lazy test
        return {'status_code': 200, 'content': request.body or ""}
    elif url.path == '/logs/profiling' and request.method == 'GET':
        # lazy test
        return {'status_code': 200, 'content': request.body or ""}

    elif url.path == '/function' and request.method == 'POST':
        return {'status_code': 200, 'content': json.dumps([5])}

    elif url.path == '/function/test' and request.method == 'GET':
        return {'status_code': 200, 'content': json.dumps(['test'])}

    elif url.path == '/execute' and request.method == 'POST':
        return {'status_code': 200, 'content': request.body or ""}

    return None


class TestQuery(unittest.TestCase):
    def __init__(self, args):
        with HTTMock(local_mock):
            self.connection = get_connection()
        unittest.TestCase.__init__(self, args)

    def test_num_tuples(self):
        with HTTMock(local_mock):
            i = rel_info(self.connection)
            self.assertEqual(i[0], 50)

    def test_partitioning(self):
        with HTTMock(local_mock):
            i = rel_info(self.connection)
            self.assertEqual(i[1].hash_partitioned,
                             RepresentationProperties().hash_partitioned)

    def test_submit(self):
        global query_request
        with HTTMock(local_mock):
            query_request = query(self.connection)
            status = self.connection.submit_query(query_request["fragments"])
            self.assertNotEqual(status, None)

    def test_execute(self):
        global query_request
        with HTTMock(local_mock):
            query_request = query(self.connection)
            status = self.connection.execute_query(query_request["fragments"])
            self.assertNotEqual(status, None)

    # TODO: fix these POST tests
    # def test_execute_program(self):
    #     global query_request
    #     with HTTMock(local_mock):
    #         query_request = query(self.connection)
    #         status = self.connection.execute_program(
    # query_request["rawQuery"])
    #         self.assertNotEquals(status, None)

    # def test_compile_program(self):
    #     global query_request
    #     with HTTMock(local_mock):
    #         query_request = query(self.connection)
    #         status = self.connection.compile_program(
    # query_request["rawQuery"])
    #         self.assertNotEquals(status, None)

    def test_get_query_plan(self):
        with HTTMock(local_mock):
            status = self.connection.get_query_plan(17, 170)
            self.assertNotEquals(status, None)

    def test_get_sent_log(self):
        with HTTMock(local_mock):
            status = self.connection.get_sent_logs(17)
            self.assertNotEquals(status, None)

    def test_get_profiling_log(self):
        with HTTMock(local_mock):
            status = self.connection.get_profiling_log(17)
            self.assertNotEquals(status, None)

    def test_reg_function(self):
        with HTTMock(local_mock):
            status = self.connection.create_function({
                'name': 'test',
                'description': 'function text',
                'outputType': 'INT_TYPE',
                'lang': FunctionTypes.PYTHON,
                'binary': "function binary"})
            print status
            self.assertNotEquals(status, None)

    def test_get_function(self):
        with HTTMock(local_mock):
            status = self.connection.get_function("test")
            self.assertNotEquals(status, None)

    def test_get_profiling_log_roots(self):
        with HTTMock(local_mock):
            status = self.connection.get_profiling_log_roots(17, 1)
            self.assertNotEquals(status, None)

    def test_upload_file(self):
        with HTTMock(local_mock):
            status = self.connection.upload_file("", [], "Hello")
            self.assertNotEquals(status, None)

    def test_validate(self):
        global query_request
        with HTTMock(local_mock):
            query_request = query(self.connection)
            plan = query_request["fragments"]
            validated = self.connection.validate_query(plan)
            self.assertNotEqual(validated, None)

    def test_query_status(self):
        global query_request
        with HTTMock(local_mock):
            query_request = query(self.connection)
            status = self.connection.get_query_status(17)
            self.assertNotEqual(status, None)

    def x_test_queries(self):
        global query_request
        with HTTMock(local_mock):
            query_request = query(self.connection)
            result = self.connection.queries()
            self.assertNotEqual(result, None)

    def test_add_udf(self):
        with HTTMock(local_mock):
            name = 'myudf'
            self.assertFalse(name in myrialparser.Parser.udf_functions)
            self.connection.create_function({'name': name,
                                             'outputType': 'STRING_TYPE'})
            self.assertTrue(name in myrialparser.Parser.udf_functions)


if __name__ == '__main__':
    unittest.main()
