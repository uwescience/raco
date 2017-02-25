import base64
import ConfigParser
import json
import csv
from time import sleep
import logging
from urlparse import urlparse, ParseResult

from raco.backends.logical import OptLogicalAlgebra
from raco.backends.myria import MyriaHyperCubeAlgebra, MyriaLeftDeepTreeAlgebra, \
    compile_to_json
from raco.backends.myria.catalog import MyriaCatalog
from raco.myrial import interpreter
from raco.myrial.parser import Parser

from .errors import MyriaError

import requests

from raco import compile, RACompiler

__all__ = ['MyriaConnection']

# String constants used in forming requests
JSON = 'application/json'
CSV = 'text/plain'
GET = 'GET'
PUT = 'PUT'
POST = 'POST'

# Enable or configure logging
logging.basicConfig(level=logging.WARN)


class FunctionTypes(object):
    POSTGRES = 0
    PYTHON = 1


class MyriaConnection(object):
    """Contains a connection the Myria REST server."""

    _DEFAULT_HEADERS = {
        'Accept': JSON,
        'Content-Type': JSON
    }

    @staticmethod
    def _parse_deployment(deployment):
        "Extract the REST server hostname and port from a deployment.cfg file"
        if deployment is None:
            return None
        config = ConfigParser.RawConfigParser(allow_no_value=True)
        config.readfp(deployment)
        master = config.get('master', '0')
        hostname = master[:master.index(':')]
        port = int(config.get('deployment', 'rest_port'))
        return (hostname, port)

    def __init__(self,
                 deployment=None,
                 hostname=None,
                 port=None,
                 ssl=False,
                 rest_url=None,
                 execution_url=None,
                 timeout=None):
        """Initializes a connection to the Myria REST server.
           (And optionally a Myria program execution URI.)

        Args:
            deployment: An open file (or other buffer) containing a
                deployment.cfg file in Myria's format. This file will be parsed
                to determine the REST server hostname and port.
            hostname: The hostname of the REST server. May be overwritten if
                deployment is provided.
            port: The port of the REST server. May be overwritten if deployment
                is provided.
            timeout: The timeout for the connection to myria.

            rest_url: a URL pointing to a Myria REST endpoint
            execution_url: a URL pointing to a Myria webserver for program
                execution
        """
        # Parse the deployment file and, if present, override the hostname and
        # port with any provided values from deployment.
        rest_config = self._parse_deployment(deployment)
        if rest_config is not None:
            hostname = hostname or rest_config[0]
            port = port or rest_config[1]
        if rest_url is not None:
            url = urlparse(rest_url)
            hostname = url.hostname
            ssl = url.scheme == "https"
            port = url.port or (443 if ssl else 80)

        if ssl:
            uri_scheme = "https"
        else:
            uri_scheme = "http"

        if execution_url is None:
            execution_url = ParseResult(scheme=uri_scheme,
                                        netloc=hostname,
                                        path="", params="",
                                        query="", fragment="").geturl()

        self._url_start = '{}://{}:{}'.format(uri_scheme, hostname, port)
        self._session = requests.Session()
        self._session.headers.update(self._DEFAULT_HEADERS)
        self.execution_url = execution_url

    def _finish_async_request(self, method, url, body=None, accept=JSON):
        headers = {
            'Accept': accept
        }
        try:
            while True:
                if '://' not in url:
                    url = self._url_start + url
                logging.info("Finish async request to {}. Headers: {}".format(
                    url, headers))
                r = self._session.request(method, url, headers=headers,
                                          data=body)
                if r.status_code in [200, 201]:
                    if accept == JSON:
                        return r.json()
                    else:
                        return r.text
                elif r.status_code in [202]:
                    # Get the new URL to poll, etc.
                    url = r.headers['Location']
                    method = GET
                    body = None
                    # Read and ignore the body
                    # response.read()
                    # Sleep 100 ms before re-issuing the request
                    sleep(0.1)
                else:
                    raise MyriaError('Error %d: %s'
                                     % (r.status_code, r.text))
        except Exception as e:
            if isinstance(e, MyriaError):
                raise
            raise MyriaError(e)

    def _make_request(self, method, url, body=None, params=None,
                      accept=JSON, get_request=False):
        headers = {
            'Accept': accept
        }
        if '://' not in url:
            url = self._url_start + url
        r = self._session.request(method, url, headers=headers,
                                  data=body, params=params, stream=True)
        logging.info("Make myria request to {}. Headers: {}".format(
                     r.url, headers))
        if r.status_code in [200, 201, 202]:
            if get_request:
                return r
            if accept == JSON:
                try:
                    return r.json()
                except ValueError, e:
                    raise MyriaError(
                        'Error %d: %s' % (r.status_code, r.text))
            else:
                return r.iter_lines()
        else:
            raise MyriaError('Error %d: %s'
                             % (r.status_code, r.text))

    def _wrap_get(self, selector, params=None, status=None, accepted=None):
        if status is None:
            status = [200]
        if accepted is None:
            accepted = []

        if '://' not in selector:
            selector = self._url_start + selector
        r = self._session.get(selector, params=params)
        if r.status_code in status:
            return r.json()
        elif r.status_code in accepted:
            return self._wrap_get(selector, params=params, status=status,
                                  accepted=accepted)
        else:
            raise MyriaError(r)

    def _wrap_post(self, selector, data=None, params=None, status=None,
                   accepted=None):
        if status is None:
            status = [201, 202]
            if accepted is None:
                accepted = [202]
        else:
            if accepted is None:
                accepted = []

        if '://' not in selector:
            selector = self._url_start + selector
        r = self._session.post(selector, data=data, params=params)
        if r.status_code in status:
            if r.headers['Location']:
                return self._wrap_get(r.headers['Location'], status=status,
                                      accepted=accepted)
            return r.json()
        else:
            raise MyriaError(r)

    def workers(self):
        """Return a dictionary of the workers"""
        return self._wrap_get('/workers')

    def workers_alive(self):
        """Return a list of the workers that are alive"""
        return self._wrap_get('/workers/alive')

    def worker(self, worker_id):
        """Return information about the specified worker"""
        return self._wrap_get('/workers/worker-{}'.format(worker_id))

    def datasets(self):
        """Return a list of the datasets that exist"""
        return self._wrap_get('/dataset')

    def dataset(self, relation_key):
        """Return information about the specified relation"""
        return self._wrap_get('/dataset/user-{}/program-{}/relation-{}'.format(
            relation_key['userName'],
            relation_key['programName'],
            relation_key['relationName']))

    def download_dataset(self, relation_key):
        """Download the data in the dataset as json"""
        return self._wrap_get('/dataset/user-{}/program-{}/relation-{}/data'
                              .format(relation_key['userName'],
                                      relation_key['programName'],
                                      relation_key['relationName']),
                              params={'format': 'json'})

    @staticmethod
    def _ensure_schema(schema):
        return {'columnTypes': schema['columnTypes'],
                'columnNames': schema['columnNames']}

    @staticmethod
    def _ensure_relation_key(relation_key):
        return {'userName': relation_key['userName'],
                'programName': relation_key['programName'],
                'relationName': relation_key['relationName']}

    def create_empty(self, relation_key, schema):
        return self.upload_source(relation_key, schema, {'dataType': 'Empty'})

    def upload_fp(self, relation_key, schema, fp):
        """Upload the data in the supplied fp to the specified user and
        relation.

        Args:
            relation_key: A dictionary containing the destination relation key.
            schema: A dictionary containing the schema,
            fp: A file pointer containing the data to be uploaded.
        """

        data = base64.b64encode(fp.read())
        source = {'dataType': 'Bytes',
                  'bytes': data}
        return self.upload_source(relation_key, schema, source)

    def upload_source(self, relation_key, schema, source):
        body = {'relationKey': self._ensure_relation_key(relation_key),
                'schema': self._ensure_schema(schema),
                'source': source}

        return self._make_request(POST, '/dataset', json.dumps(body))

    def execute_program(self, program, language="MyriaL", server=None):
        """Execute the program in the specified language on Myria, polling
        its status until the query is finished. Returns the query status
        struct.

        Args:
            program: a Myria program as a string.
            language: the language in which the program is written
                      (default: MyriaL).
        """

        body = {"query": program, "language": language}
        r = requests.post((server or self.execution_url) + '/execute',
                          data=body)
        if r.status_code != 201:
            raise MyriaError(r)

        query_uri = r.json()['url']
        while True:
            r = requests.get(query_uri)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 202:
                # Sleep 100 ms before re-checking the status
                sleep(0.1)
                continue
            raise MyriaError(r)

    def compile_program(self, program, language="MyriaL", **kwargs):
        """Get a compiled plan for a given program.

        Args:
            program: a Myria program as a string.
            language: the language in which the program is written
                      (default: MyriaL).
        """
        logical = self._get_plan(program, language, 'logical', **kwargs)
        physical = self._get_plan(program, language, 'physical', **kwargs)
        compiled = compile_to_json(program, logical, physical, language)
        compiled['profilingMode'] = ["QUERY", "RESOURCE"] \
            if kwargs.get('profile', False) else []
        return compiled

    def submit_query(self, query):
        """Submit the query to Myria, and return the status including the URL
        to be polled.

        Args:
            query: a Myria physical plan as a Python object.
        """

        body = json.dumps(query)
        return self._wrap_post('/query', data=body)

    def execute_query(self, query):
        """Submit the query to Myria, and poll its status until it finishes.

        Args:
            query: a Myria physical plan as a Python object.
        """

        body = json.dumps(query)
        return self._finish_async_request(POST, '/query', body)

    def validate_query(self, query):
        """Submit the query to Myria for validation only.

        Args:
            query: a Myria physical plan as a Python object.
        """

        body = json.dumps(query)
        return self._make_request(POST, '/query/validate', body)

    def get_query_status(self, query_id):
        """Get the status of a submitted query.

        Args:
            query_id: the id of a submitted query
        """

        resource_path = '/query/query-%d' % int(query_id)
        return self._make_request(GET, resource_path)

    def get_query_plan(self, query_id, subquery_id):
        """Get the saved execution plan for a submitted query.

        Args:
            query_id: the id of a submitted query
            subquery_id: the subquery id within the specified query
        """

        resource_path = '/query/query-{q}/subquery-{s}'.format(
            q=int(query_id), s=int(subquery_id))
        return self._make_request(GET, resource_path)

    def get_fragment_ids(self, query_id, worker_id):
        """Get the number of fragments in a query plan.

        Args:
            query_id: the id of a submitted query
            worker_id: the id of a worker
        """
        status = self.get_query_status(query_id)
        if 'fragments' in status['plan']:
            fids = []
            for fragment in status['plan']['fragments']:
                if int(worker_id) in map(int, fragment['workers']):
                    fids.append(fragment['fragmentIndex'])
            return fids
        else:
            return []

    def get_sent_logs(self, query_id, fragment_id=None):
        """Get the logs for where data was sent.

        Args:
            query_id: the id of a submitted query
            fragment_id: the id of a fragment
        """
        resource_path = '/logs/sent?queryId=%d' % int(query_id)
        if fragment_id is not None:
            resource_path += '&fragmentId=%d' % int(fragment_id)
        response = self._make_request(GET, resource_path, accept=CSV)
        return csv.reader(response)

    def get_profiling_log(self, query_id, fragment_id=None):
        """Get the logs for operators.

        Args:
            query_id: the id of a submitted query
            fragment_id: the id of a fragment
        """
        resource_path = '/logs/profiling?queryId=%d' % int(query_id)
        if fragment_id is not None:
            resource_path += '&fragmentId=%d' % int(fragment_id)
        response = self._make_request(GET, resource_path, accept=CSV)
        return csv.reader(response)

    def get_profiling_log_roots(self, query_id, fragment_id):
        """Get the logs for root operators.

        Args:
            query_id: the id of a submitted query
            fragment_id: the id of a fragment
        """
        resource_path = '/logs/profilingroots?queryId=%d&fragmentId=%d' % (int(
            query_id), int(fragment_id))
        response = self._make_request(GET, resource_path, accept=CSV)
        return csv.reader(response)

    def queries(self, limit=None, max_id=None, min_id=None, q=None):
        """Get count and information about all submitted queries.

        Args:
            limit: the maximum number of query status results to return.
            max_id: the maximum query ID to return.
            min_id: the minimum query ID to return. Ignored if max_id is
                    present.
            q: a text search for the raw query string.
        """
        params = {}
        if limit is not None:
            params['limit'] = limit
        if max_id is not None:
            params['max'] = max_id
        if min_id is not None:
            params['min'] = min_id
        if q is not None:
            params['q'] = q

        resource_path = '/query'
        r = self._make_request(GET, resource_path, params=params,
                               get_request=True)
        return r.json()

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

        from requests_toolbelt import MultipartEncoder

        fields = [('relationKey', relation_key), ('schema', schema),
                  ('overwrite', overwrite), ('delimiter', delimiter),
                  ('binary', binary), ('isLittleEndian', is_little_endian)]
        fields = [(name, (name, json.dumps(value), 'application/json'))
                  for (name, value) in fields]
        # data must be last
        if binary:
            data_type = 'application/octet-stream'
        else:
            data_type = 'text/plain'
        fields.append(('data', ('data', data, data_type)))

        m = MultipartEncoder(fields=fields)
        r = self._session.post(self._url_start + '/dataset', data=m,
                               headers={'Content-Type': m.content_type})
        if r.status_code not in (200, 201):
            raise MyriaError('Error %d: %s'
                             % (r.status_code, r.text))
        return r.json()

    def get_function(self, name):
        """ Get user defined function metadata """
        return self._wrap_get('/function/{}'.format(name))

    def create_function(self, d):
        """Register a User Defined Function with Myria """
        return self._make_request(POST, '/function', json.dumps(d))

    def get_functions(self):
        """ List all the user defined functions in Myria """
        return self._wrap_get('/function')

    def _get_plan(self, query, language, plan_type, **kwargs):
        catalog = MyriaCatalog(self)
        algebra = MyriaHyperCubeAlgebra(catalog) \
            if kwargs.get('multiway_jon', False) \
            else MyriaLeftDeepTreeAlgebra()

        if language.lower() == "datalog":
            return self._get_datalog_plan(query, plan_type, algebra, **kwargs)
        elif language.lower() in ["myrial", "sql"]:
            return self._get_myrial_or_sql_plan(query, plan_type,
                                                catalog, algebra, **kwargs)
        else:
            raise NotImplementedError('Language %s not supported' % language)

    @staticmethod
    def _get_myrial_or_sql_plan(query, plan_type, catalog, algebra, **kwargs):
        parsed = Parser().parse(query)
        processor = interpreter.StatementProcessor(catalog)
        processor.evaluate(parsed)

        if plan_type == 'logical':
            return processor.get_physical_plan(
                target_alg=OptLogicalAlgebra())
        elif plan_type == 'physical':
            return processor.get_physical_plan(
                target_alg=algebra,
                multiway_join=kwargs.get('multiway_join', False),
                push_sql=kwargs.get('push_sql', True))
        else:
            raise NotImplementedError('Myria plan type %s' % plan_type)

    @staticmethod
    def _get_datalog_plan(query, plan_type, algebra, **kwargs):
        datalog = RACompiler()
        datalog.fromDatalog(query)

        if not datalog.logicalplan:
            raise SyntaxError("Unable to parse Datalog")
        elif plan_type == 'logical':
            return datalog.logicalplan
        elif plan_type == 'physical':
            datalog.optimize(target=algebra,
                             push_sql=kwargs.get('push_sql', True))
            return datalog.physicalplan
        else:
            raise NotImplementedError('Datalog plan type %s' % plan_type)
