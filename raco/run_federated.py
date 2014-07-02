from raco.compile import optimize
from raco.language import FederatedAlgebra
from raco.algebra import LogicalAlgebra, Sequence
from raco.federatedlang import *

from scidbpy import connect

def run(logical_plan):
    seq_op = optimize(logical_plan, target=FederatedAlgebra(),
                      source=LogicalAlgebra)
    assert isinstance(seq_op, Sequence)

    for op in seq_op.args:
        if isinstance(op, RunAQL):
            print op.command
            sdb = connect(op.connection)
            sdb._execute_query(op.command)
