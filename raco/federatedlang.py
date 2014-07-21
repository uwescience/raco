from raco import algebra
from raco import rules
from raco.language import Language
from raco.language import MyriaLeftDeepTreeAlgebra
from raco.compile import optimize
from raco.myrialang import compile_to_json


class Federated(Language):
    pass


class FederatedOperator(algebra.ZeroaryOperator):
    language = Federated

    def shortStr(self):
        return repr(self)

    def scheme(self):
        raise NotImplementedError()

class Runner(FederatedOperator):
    def __init__(self, command, connection=None):
        self.command = command
        self.connection = connection


class RunAQL(Runner):
    """Run an AQL query on a SciDB instance specified by the programmer"""
    def __repr__(self):
        return "RunAQL(%s, %s)" % (self.command, self.connection)

class RunMyria(Runner):
    """Run a Myria query on the UW cluster"""

    def __repr__(self):
        return "RunMyria(%s, %s)" % (self.command, self.connection)

dispatchmap = {"aql": RunAQL, "myria": RunMyria, "afl": RunAQL}


class Dispatch(rules.Rule):
    def fire(self, expr):
        if isinstance(expr, algebra.Sequence):
            return expr  # Retain top-level sequence operator
        if isinstance(expr, algebra.ExportMyriaToScidb):
            return expr
        if isinstance(expr, algebra.ExecScan):
            # Some kind of custom code that we must pass through
            return dispatchmap[expr.languagetag](expr.command, expr.connection)
        else:
            # Just a logical plan that we will dispatch to Myria by default
            pp = optimize(expr, target=MyriaLeftDeepTreeAlgebra(),
                          source=algebra.LogicalAlgebra)
            json = compile_to_json("raw query", "logical plan", pp)
            return dispatchmap["myria"](json)


class FederatedAlgebra(object):
    language = Federated

    operators = [RunAQL, RunMyria]

    def opt_rules(self):
        return [Dispatch()]
