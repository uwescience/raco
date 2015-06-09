from raco.compile import optimize
from raco.language.federatedlang import FederatedAlgebra
from raco.algebra import Sequence, ExportMyriaToScidb
from raco.language.federatedlang import *
from raco.language.logical import OptLogicalAlgebra

import logging
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
    pure_myria_query = all([isinstance(x, RunMyria) for x in seq_op.args])

    logging.info("Running federated query: pure_myra=%s" % pure_myria_query)

    for op in seq_op.args:
        if isinstance(op, RunAQL):
            logging.info("Running scidb query: %s" % op.command)
            sdb = scidb_conn_factory.connect(op.connection)
            try:
                sdb._execute_query(op.command)
            except Exception as ex:
                logging.info("SciDB query failure: " + str(ex))
        elif isinstance(op, RunMyria):
            logging.info("Running myria query...")

            res = myria_conn.submit_query(op.command)
            _id = res['queryId']
            if not pure_myria_query:
                outs.append(wait_for_completion(myria_conn, _id))
            else:
                return res
        elif isinstance(op, ExportMyriaToScidb):
            logging.info("Exporting to scidb...")

            # Download Myria data -- assumes query has completed
            relk = op.myria_relkey
            key = {'userName': relk.user, 'programName': relk.program,
                   'relationName': relk.relation}
            dataset = myria_conn.download_dataset(key)

            logging.info("Finished myria download; starting scidb upload")

            # Unpack first result
            vals = [v for row in dataset for v in row.values()]
            val0 = vals[0]

            # Store as scidb array
            sdb = scidb_conn_factory.connect(op.connection)
            try:
                sdb.query("remove(%s)" % op.scidb_array_name)
            except:
                pass

            sdb.query("store(build(<RecordName:int64>[i=0:0,1,0], %d), %s)" % (
                val0, op.scidb_array_name))

    logging.info("Returning from federated query")

    if len(outs) > 0:
        return outs[-1]  # XXX This is strange
