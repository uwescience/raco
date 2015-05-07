
'''
A notional raco-only script for executing a MyriaL query
'''

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
physical_plan = raco.optimize(logical_plan
                             , target=MyriaAlgebra
                             , catalog=catalog
                             , rules=MyriaAlgbra.rules
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
  
  
