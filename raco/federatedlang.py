from raco import algebra
from raco import rules
from raco.language import Language
from raco.language import MyriaAlgebra
from raco.compile import optimize
import raco.myrialang

class Federated(Language):
    pass

class FederatedOperator(object):
    language = Federated

class Runner(FederatedOperator, algebra.ExecScan):
    def __init__(self, command, connection):
      self.command = command
      self.connection = connection

    def __str__(self):
      return "%s on %s" % (self.command, self.connection)

class RunAQL(Runner):
    """Run an AQL query on a SciDB instance specified by the programmer"""
    pass

class RunMyriaAtUW(Runner):
    """Run a Myria query on the UW cluster"""
    def __init__(self, command):
      self.command = command
      self.connection = None # Hardcode the connection here

class RunSQL(Runner):
    """Run a SQL query on a given DB"""
    pass

class MoveSciDBToMyria(FederatedOperator):
  pass

class MoveMyriaToSciDB(FederatedOperator):
  pass

dispatchmap = {"aql" : RunAQL
              ,"sql" : RunSQL
              ,"myria" : RunMyriaAtUW 
              }

class Dispatch(rules.Rule):
    def fire(self, expr):
        if isinstance(expr, algebra.ExecScan):
            # Some kind of custom code that we must pass through
            return dispatchmap[expr.languagetag](expr.command, expr.connection)
        else:
            # Just a logical plan that we will dispatch to Myria by default
            pps = optimize([(None, expr)]
                          ,target=MyriaAlgebra
                          ,source=algebra.LogicalAlgebra
                          )
            json = raco.myrialang.compile_to_json("no query", pps[0][1], pps[0][1])
            return dispatchmap["myria"](json)


class FederatedAlgebra(object):
    language = Federated

    operators = [ RunAQL
                , RunMyriaAtUW
                ]
    rules = [Dispatch()
            ]
