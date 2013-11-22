from raco import RACompiler
import raco.algebra as alg
import random
#plan...
#if I'm a unary operator, set edge to child, evaluate child
#if I'm a binary operator, set edge to left child, evaluate left child, set edge to right child, evaluate right child

class unique_id :
    def __init__(self) :
        self.seed = random.getrandbits(32)

    def next_id(self) :
        self.seed += 1
        return self.seed

#run as:
#query = <query>
#parsedprogram = parse(query)
#exprs = parsedprogram.toRA()
#generateDot(exprs,<file_name>)
# bash ~: dot -Tps <file_name> -o graph.ps
def generateDot(inputs,output_file) :
    result = inputs[0][0]
    x = inputs[0][1]
    u = unique_id()
    ID = u.next_id()
    s=['digraph G {',result + '-> ' + str(ID) + ';']
    #s = 'digraph G {\n'

    generateDotRecursive(x,s,ID,u)
    #s += '}\n'
    #print '}'
    s.append('}')
    open(output_file,'w').write('\n'.join(s))

def generateDotRecursive(x,s,ID,u) :
    if isinstance(x,alg.ZeroaryOperator) :
        s.append(str(ID) + '[label = "' + str(x) + '"];')
        #s += str(ID) + '[label = "' + str(s) + '"];\n'
        #print str(ID) + '[label = "' + str(x) + '"];'

    elif isinstance(x,alg.UnaryOperator) :
        child = u.next_id()
        #s += str(ID) + '->' + str(child) + ';\n'
        #print str(ID) + '->' + str(child) + ';'
        s.append(str(ID) + '->' + str(child) + ';')
        #s += str(ID) + '[label = "' + x.opname() + '"];\n'
        #print str(ID) + '[label = "' + x.opname() + '"];'
        if isinstance(x,alg.Project) :
             s.append(str(ID) + '[label = "' + x.opname() + ' ' + str(x.columnlist)+ '"];')
        elif isinstance(x,alg.Select) :
            s.append(str(ID) + '[label = "' + x.opname() + ' ' + str(x.condition)+ '"];')
        else :
            s.append(str(ID) + '[label = "' + x.opname() + '"];')
        generateDotRecursive(x.input,s,child,u)

    elif isinstance(x,alg.BinaryOperator) :
        l = u.next_id()
        #s += str(ID) + '->' + str(l) + ';\n'
        #print str(ID) + '->' + str(l) + ';'
        s.append(str(ID) + '->' + str(l) + ';')
        generateDotRecursive(x.left,s,l,u)
        r = u.next_id()
        #s += str(ID) + '->' + str(r) + ';\n'
        #print str(ID) + '->' + str(r) + ';'
        s.append(str(ID) + '->' + str(r) + ';')
        generateDotRecursive(x.right,s,r,u)

        #s += str(ID) + '[label = "' + x.opname() + '"];\n'
        #print str(ID) + '[label = "' + x.opname() + '"];'
        if isinstance(x,alg.Join) :
            s.append(str(ID) + '[label = "' + x.opname() + ' ' + str(x.condition)+ '"];')
        else :
            s.append(str(ID) + '[label = "' + x.opname() + '"];')

    else :
        #not handling n-ary operators yet
        print 'ERROR: unknown operator'
