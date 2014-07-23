from raco.compile import optimize
from raco.language import FederatedAlgebra
from raco.algebra import LogicalAlgebra, Sequence, ExportMyriaToScidb
from raco.federatedlang import *

import numpy as np
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
    async = all([True for x in seq_op.args if isinstance(x, RunMyria)])

    for op in seq_op.args:
        if isinstance(op, RunAQL):
            sdb = scidb_conn_factory.connect(op.connection)
            sdb._execute_query(op.command)
        elif isinstance(op, RunMyria):
            res = myria_conn.submit_query(op.command)
            _id = res['queryId']
            if not async:
                outs.append(wait_for_completion(myria_conn, _id))
            else:
                outs.append(res)
        elif isinstance(op, ExportMyriaToScidb):
            # Download Myria data -- assumes query has completed
            relk = op.myria_relkey
            key = {'userName': relk.user, 'programName': relk.program,
                   'relationName': relk.relation}
            dataset = myria_conn.download_dataset(key)

            # Unpack data into a 1-dimensional array
            # XXX non-sensical behavior for multi-column relations
            vals = [v for row in dataset for v in row.values()]
            A = np.array(vals)

            # Store as scidb array
            sdb = scidb_conn_factory.connect(op.connection)
            if op.scidb_array_name in sdb.list_arrays():
                sdb.query("remove(%s)" % op.scidb_array_name)
            Asdb = sdb.from_array(A)
            Asdb.rename(op.scidb_array_name, persistent=True)

    if len(outs) > 0:
        return outs[-1]  # XXX This is strange
