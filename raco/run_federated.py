from raco.compile import optimize
from raco.language import FederatedAlgebra
from raco.algebra import LogicalAlgebra, Sequence, ExportMyriaToScidb
from raco.federatedlang import *

import time

def wait_for_completion(conn, query_id, repeats=10):
    for i in range(repeats):
        res = conn.get_query_status(query_id)
        status = res['status']
        print status
        if status == 'SUCCESS':
            return res
        time.sleep(1)

    raise Exception("Query failure")

def run(logical_plan, myria_conn, scidb_conn_factory):
    seq_op = optimize(logical_plan, target=FederatedAlgebra(),
                      source=LogicalAlgebra)
    assert isinstance(seq_op, Sequence)

    outs = []
    for op in seq_op.args:
        if isinstance(op, RunAQL):
            sdb = scidb_conn_factory.connect(op.connection)
            sdb._execute_query(op.command)
        elif isinstance(op, RunMyria):
            res = myria_conn.submit_query(op.command)
            _id = res['queryId']
            outs.append(wait_for_completion(myria_conn, _id))
        elif isinstance(op, ExportMyriaToScidb):
            pass
            # Fetch result schema
            # Fetch result
            # Store as scidb array

    if len(outs) > 0:
        return outs[-1]  # XXX This is strange
