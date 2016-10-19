import logging

import sys

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TCompactProtocol

#PYTHONPATH=$ACCUMULO_HOME/proxy/thrift/gen-py/accumulo
from accumulo import AccumuloProxy
from accumulo.ttypes import *

logging.basicConfig(level=logging.WARN)


class AccumuloConnection(object):

    def __init__(self,
                 proxy_host='localhost',
                 proxy_port=42424,
                 accumulo_user='root',
                 accumulo_password='secret'):
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
        transport = TSocket.TSocket('localhost', 42424)
        transport = TTransport.TFramedTransport(transport)
        protocol = TCompactProtocol.TCompactProtocol(transport)
        self.client = AccumuloProxy.Client(protocol)
        transport.open()
        self.login = self.client.login('root', {'password': 'secret'})
        print self.client.listTables(self.login)

    def getTableProperties(self, table):
        return self.client.getTableProperties(self.login, table)

