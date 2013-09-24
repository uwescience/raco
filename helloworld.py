from raco import RACompiler
from raco.language import MyriaAlgebra
from raco.myrialang import compile_to_json
import json

def json_pretty_print(dictionary):
    """a function to pretty-print a JSON dictionary.
From http://docs.python.org/2/library/json.html"""
    return json.dumps(dictionary, sort_keys=True, 
            indent=2, separators=(',', ': '))

# A simple join
join = """
A(x,z) :- Twitter(x,y), Twitter(y,z)
"""
# A multi-join version
multi_join = """
A(x,w) :- R3(x,y,z), S3(y,z,w)
"""

# Triangles
triangles = """
A(x,y,z) :- R(x,y),S(y,z),T(z,x)
"""

# Three hops
three_hops = """
ThreeHops(x,w) :- TwitterK(x,y),TwitterK(y,z),TwitterK(z,w)
"""

# Which one do we use?
query = join

def comment(s):
  print "/*\n%s\n*/" % str(s)

# Create a cmpiler object
dlog = RACompiler()

# parse the query
dlog.fromDatalog(query)
print "************ LOGICAL PLAN *************"
cached_logicalplan = str(dlog.logicalplan)
print dlog.logicalplan
print

# Optimize the query, includes producing a physical plan
print "************ PHYSICAL PLAN *************"
dlog.optimize(target=MyriaAlgebra, eliminate_common_subexpressions=False)
print dlog.physicalplan
print

# generate code in the target language
print "************ CODE *************"
myria_json = compile_to_json(query, cached_logicalplan, dlog.physicalplan)
print json_pretty_print(myria_json)
print

# dump the JSON to output.json
print "************ DUMPING CODE TO output.json *************"
with open('output.json', 'w') as outfile:
    json.dump(myria_json, outfile)

