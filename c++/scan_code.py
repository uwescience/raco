from raco import RACompiler
import raco.algebra as alg
from collections import deque

def gen_code_for_scan(varname,filename,numcols,scan_count) :
    code = open('scan.template').read()
    code = code.replace('$$varname$$',varname)
    code = code.replace('$$filename$$','"' + str(filename) + '"')
    code = code.replace('$$numcolumns$$',str(numcols))
    code = code.replace('$$tmp_vector$$','tmp_vector' + str(scan_count))
    code = code.replace('$$f$$','f' + str(scan_count))
    code = code.replace('$$count$$', 'count' + str(scan_count))
    return code

def generate_header(query_name) :
    code = open('header.template').read()
    code = code.replace('$$qn$$',query_name)
    return code

def generate_files(tree, query_name) :
    f = open(query_name + '.h','w')
    f.write(generate_header(query_name))
    f.close()

    f = open(query_name + '.cpp','w')
    f.write('#include "' + query_name + '.h"\n\n')
    f.write('void query () {\n')

    #walk the tree
    #if we find a scan, generate code
    scans = 0
    nodes = deque()
    nodes.append(tree)

    while len(nodes) != 0 :
        n = nodes.pop()
        if isinstance(n,alg.ZeroaryOperator) :
            if isinstance(n,alg.Scan) :
                scans += 1
                f.write(gen_code_for_scan(n.relation.name,n.relation.name,len(n.relation.scheme),scans))
            #else :
                #nothing?
        elif isinstance(n,alg.UnaryOperator) :
            nodes.append(n.input)
        elif isinstance(n,alg.BinaryOperator) :
            nodes.append(n.right)
            nodes.append(n.left)

    f.write('}\n\nint main() { query(); }')
