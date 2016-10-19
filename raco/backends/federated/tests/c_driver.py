from raco.backends.cpp.cpp import CCAlgebra
from raco.backends.federated.connection import FederatedConnection
from raco.backends.logical import OptLogicalAlgebra
from raco.backends.myria.connection import MyriaConnection
from raco.backends.myria.catalog import MyriaCatalog
from raco.backends.myria import MyriaLeftDeepTreeAlgebra
from raco.backends.federated.catalog import FederatedCatalog
from raco.backends.federated.algebra import FederatedAlgebra
from raco.catalog import FromFileCatalog
from raco.compile import compile
import raco.myrial.interpreter as interpreter
import raco.myrial.parser as myrialparser
import os

masterHostname = os.environ.get('sparkurl', 'localhost')
def get_myria_connection():
    execution_url = os.environ.get('MYRIAX_REST_HOST', 'localhost')
    connection = MyriaConnection(hostname=execution_url, port=8753)
    return connection

program_fquery="""
NF = scan(netflow);
NFSUB = select SrcAddr as src_ip, DstAddr as dst_ip, 1.0 as value from NF where TotBytes > 5120;
DNS = scan('/Users/shrainik/Dropbox/raco/examples/fed_accumulo_spark_c/dnssample_parsed.txt');
graph = select d1.dns as row, d2.dns as col, n.value from NFSUB n, DNS d1, DNS d2
    where n.src_ip = d1.ip and n.dst_ip = d2.ip;
gammas = select a.row as u, b.row as v, count(b.value) as gamma from graph a, graph b where a.col == b.col;
out_d = select row, count(value) as od from graph;
J = select a.u as src_name, a.v as dst_name, a.gamma/(b.od + c.od - a.gamma) as jaccard_coeff from gammas a, out_d b, out_d c where a.u = b.row and a.v = c.row;

store(J, nameJaccard);
"""
program_fquery_simple ="""
NF = scan(netflow);
NFSUB = select SrcAddr as src_ip, DstAddr as dst_ip, 1.0 as value from NF where TotBytes > 5120;
DNS = scan('/Users/shrainik/Dropbox/raco/examples/fed_accumulo_spark_c/dnssample_parsed.txt');
graph = select d1.dns as row, d2.dns as col, n.value from NFSUB n, DNS d1, DNS d2
    where n.src_ip = d1.ip and n.dst_ip = d2.ip;
store(graph, ipGraph);
"""
myriaconn = get_myria_connection()
myriacatalog = MyriaCatalog(myriaconn)

catalog = FromFileCatalog.load_from_file(os.path.join(os.path.dirname('/Users/shrainik/Dropbox/raco/examples/'), 'catalog.py'))
catalog = FederatedCatalog([myriacatalog, catalog])

parser = myrialparser.Parser()
processor = interpreter.StatementProcessor(catalog, True)
statement_list = parser.parse(program_fquery)
processor.evaluate(statement_list)

algebras = [OptLogicalAlgebra(), MyriaLeftDeepTreeAlgebra(), CCAlgebra()]
falg = FederatedAlgebra(algebras, catalog, crossproducts=False)

federated_plan = processor.get_physical_plan(target_alg=falg)
fed_conn = FederatedConnection([myriaconn])
fed_conn.execute_query(federated_plan)