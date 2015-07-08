#!/usr/bin/env python

'''
A notional raco-only script for executing a MyriaL query
'''

assert False, "This script is currently only an example of a new interface yet to be implemented"

import raco

from raco.convenience import executeMyriaQuery
from raco.op.algebra import LogicalAlgebra

# Choose a backend target system
from raco.backend.myriaX import MyriaConnection, MyriaAlgebra
connection = myriaX.MyriaConnection(hostname='vega.cs.washington.edu', port=1776)

# Choose a frontend language
from raco.inputlanguage import myriaL
processor = myriaL.MyrialInterpreter.StatementProcessor(catalog)

# write the query
query = '''
T1 = scan(TwitterK);
X = [from T1 emit T1.$0 where T1.$0 < 100];
store(X, myresult);
'''

# parse the query to produce a logical plan
logical_plan = processor.parse(query)

# manhandle the logical plan here, if desired.

# from this point on, we don't care we the logical plan came from

# get a catalog object 
# (we could also create one from a local file, or use a dummy catalog of some kind)
catalog = connection.catalog() 

# optimize the query to produce a physical plan
# We do this in three steps: Logical -> Logical, Logical->Parallel, Parallel->Myria
# We can have any number of IRs we wish.
# Each IR can provide its own rules and cost function, or we can assume defaults
# source is required, we can provide defaults for the other arguments
# defaul catalog will just accept any consistent query
# default rules will be the rules of the target Algebra
# default costfunction will be the costfunction of the target Algebra
#   (if no cost function is provided, maybe just apply rules greedily?)
optimized_logical = raco.optimize(logical_plan
                                 , target=LogicalAlgebra
                                 , catalog=catalog
                                 , rules=LogicalAlgebra.rules
                                 , costfunction=LogicalAlgebra.costfunction
                                 )
                             
physical_plan = raco.optimize(optimized_local
                             , target=MyriaAlgebra
                             , catalog=catalog
                             , rules=MyriaAlgebra.rules
                             , costfunction=MyriaAlgebra.costfunction
                             )

# manhandle with the physical plan here, if desired

# compile the physical plan
compiled = physical_plan.compile()

# manhandle the json plan here, if desired

# execute the query
queryid = connection.submit_query(compiled)
while True:
  status = connection.status(queryid)
  print status
  if connection.ready(status):
    break

# get the result, perhaps limited to 10000 records or something.
result = connection.fetch("myresult")
  
  

# Or if you just want to connect to Myria and get the result of a query in a few lines
# ie should MyriaXConnection have default settings for calling raco.optimize?
connection = myriaX.MyriaConnection(hostname='vega.cs.washington.edu', port=1776)
query = '''
T1 = scan(TwitterK);
X = [from T1 emit T1.$0 where T1.$0 < 100];
store(X, myresult);
'''
result = executeMyriaQuery(connection, query)
