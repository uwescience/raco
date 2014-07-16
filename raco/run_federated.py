from raco.compile import optimize
from raco.language import FederatedAlgebra
from raco.algebra import LogicalAlgebra, Sequence
from raco.federatedlang import *


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
            outs.append(myria_conn.submit_query(op.command))

    if len(outs) > 0:
        return outs[-1]  # XXX This is strange
