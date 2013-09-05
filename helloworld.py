from raco import RACompiler
from raco.language import MyriaAlgebra
from raco.algebra import LogicalAlgebra, gensym, ZeroaryOperator, UnaryOperator, BinaryOperator, NaryOperator
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

# Which one do we use?
query = join

def comment(s):
  print "/*\n%s\n*/" % str(s)

# Create a cmpiler object
dlog = RACompiler()

# parse the query
dlog.fromDatalog(query)
print "************ LOGICAL PLAN *************"
print dlog.logicalplan
print

# Optimize the query, includes producing a physical plan
print "************ PHYSICAL PLAN *************"
dlog.optimize(target=MyriaAlgebra, eliminate_common_subexpressions=False)
print dlog.physicalplan
print

# generate code in the target language
print "************ CODE *************"
phys = dlog.physicalplan

syms = {}

def one_fragment(rootOp):
    cur_frag = [rootOp]
    if id(rootOp) not in syms:
        syms[id(rootOp)] = gensym()
    queue = []
    if isinstance(rootOp, MyriaAlgebra.fragment_leaves):
        for child in rootOp.children():
            queue.append(child)
    else:
        for child in rootOp.children():
            (child_frag, child_queue) = one_fragment(child)
            cur_frag += child_frag
            queue += child_queue
    return (cur_frag, queue)

def fragments(rootOp):
    queue = [rootOp]
    ret = []
    while len(queue) > 0:
        rootOp = queue.pop(0)
        (op_frag, op_queue) = one_fragment(rootOp)
        ret.append(reversed(op_frag))
        queue.extend(op_queue)
    return ret

def call_compile_me(op):
    opsym = syms[id(op)]
    childsyms = [syms[id(child)] for child in op.children()]
    if isinstance(op, ZeroaryOperator):
        return op.compileme(opsym)
    if isinstance(op, UnaryOperator):
        return op.compileme(opsym, childsyms[0])
    if isinstance(op, BinaryOperator):
        return op.compileme(opsym, childsyms[0], childsyms[1])
    if isinstance(op, NaryOperator):
        return op.compileme(opsym, childsyms)
    raise NotImplementedError("unable to handle operator of type "+type(op))

all_frags = []
for (label, rootOp) in phys:
    syms[id(rootOp)] = label
    label, rootOp
    frags = fragments(rootOp)
    all_frags.extend([{'operators': [call_compile_me(op) for op in frag]} for frag in frags])
    syms.clear()

query = {
        'fragments' : all_frags,
        'raw_datalog' : query,
        'logical_ra' : str(dlog.logicalplan)
        }
print json_pretty_print(query)
print

# dump the JSON to output.json
print "************ DUMPING CODE TO output.json *************"
with open('output.json', 'w') as outfile:
    json.dump(query, outfile)

