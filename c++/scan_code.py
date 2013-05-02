from raco import RACompiler
import raco.algebra as alg
import raco.clang as clang
import raco.boolean as rbool
from collections import deque
   
class cpp_code :

    #constructor
    def __init__(self,physicalplan,queryname) :
        #physical plan is a list of tuples, each containing a str and a clang operator
        self.plan = physicalplan
        self.cpp_code = ''
        self.query_name = queryname
        self.scan_count = 0
        self.node_to_name = {}
        self.node_to_hash = {}
        self.indent = 0

    #generate code call
    def gen_code(self) :
        f = open(self.query_name + '.h','w')
        f.write(self.generate_header(self.query_name))
        f.close()

        self.cpp_code += '#include "' + self.query_name + '.h"\n\n'
        self.cpp_code += 'void query () {\n'
        self.indent += 4

        #generate_files call
        for s,c in self.plan :
            self.visit(c)
        self.cpp_code +='\n}\n\nint main() { query(); }'
        self.indent -= 4

        f = open(self.query_name + '.cpp','w')
        f.write(self.cpp_code)
        f.close()

    #load scan template
    def gen_code_for_scan(self,n) :
        varname = n.relation.name
        filename = n.relation.name #could be changed later
        numcols = len(n.relation.scheme)
        code = open('scan.template').read()
        code = code.replace('$$varname$$',varname)
        code = code.replace('$$filename$$','"' + str(filename) + '"')
        code = code.replace('$$numcolumns$$',str(numcols))
        code = code.replace('$$tmp_vector$$','tmp_vector' + str(self.scan_count))
        code = code.replace('$$f$$','f' + str(self.scan_count))
        code = code.replace('$$count$$', 'count' + str(self.scan_count))
        self.node_to_name[n] = varname
        self.scan_count += 1
        return code.replace('\n','\n' + ' '*self.indent)

    #code to generate header
    def generate_header(self,query_name) :
        code = open('header.template').read()
        code = code.replace('$$qn$$',query_name)
        return code.replace('\n','\n' + ' '*self.indent)

    def generate_hash_code(self,hashname,relation,column) :
        code = open('hash.template').read()
        code = code.replace('$$hashname$$',hashname)
        code = code.replace('$$relation$$',relation)
        code = code.replace('$$column$$',str(column))
        return code.replace('\n','\n' + ' '*self.indent)

    #messy code for generating join chain code
    def generate_join_chain(self,n) :
        #step 1: update columns in join conditions
        
        for arg in n.args :
            print arg
        print '---'

        for c in n.joinconditions :
            print c 
        '''
        tot = 0
        for i in range(0,len(n.joinconditions)) :
            pos = n.joinconditions[i].left.position
            print n.joinconditions[i]
            for arg in n.args :
                if pos >= len(arg.relation.scheme) :
                    pos -= len(arg.relation.scheme)
            n.joinconditions[i].left.position = pos

        print '---'

        for c in n.joinconditions :
            print c
        '''

        #step 2: create necessary hashes
        for i in range(0,len(n.joinconditions)) :
            pos = n.joinconditions[i].right.position
            node = n.args[i+1]
            name = node.relation.name + str(pos) + '_hash'
            self.cpp_code += self.generate_hash_code(name,node.relation.name,pos)
            self.node_to_hash[node] = name

        #step 3: nasty code generation
            
        return

    #recursive code to walk the tree
    def visit (self,n):
        #generate code for a scan
        if isinstance(n,alg.ZeroaryOperator) :
            if isinstance(n,alg.Scan) :
                self.cpp_code += self.gen_code_for_scan(n)
            #else :
                #nothing?

        #nothing here yet
        elif isinstance(n,alg.UnaryOperator) :
            self.visit(n.input)

        #nothing here yet
        elif isinstance(n,alg.BinaryOperator) :
            self.visit(n.right)
            self.visit(n.left)

        #generate code for a join chain
        elif isinstance(n,alg.NaryOperator) :
            if isinstance(n,clang.FilteringNLJoinChain) :
                for arg in n.args :
                    self.visit(arg)
                self.generate_join_chain(n)

            

