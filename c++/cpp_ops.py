from raco import RACompiler
import raco.algebra as alg

class VectorScan :
    #logical scan is a Scan object from alg
    def __init__(self,logical_scan) :
        self.logic = logical_scan

    def code(self,scan_count) :
        varname = logic.relation.name
        filename = logic.relation.name #for now...
        numcols = len(logic.relation.scheme)
        code = open('scan.template').read()
        code = code.replace('$$varname$$',varname)
        code = code.replace('$$filename$$','"' + str(filename) + '"')
        code = code.replace('$$numcolumns$$',str(numcols))
        code = code.replace('$$tmp_vector$$','tmp_vector' + str(scan_count))
        code = code.replace('$$f$$','f' + str(scan_count))
        code = code.replace('$$count$$', 'count' + str(scan_count))
        return code

class HashForJoin :
    def __init__(self) : #later this will take some sort of algebra object
        self.hashname = ""
        self.relation = ""
        self.column = 0

    def code(self) :
        code = open('hash.template').read()
        code = code.replace('$$hashname$$',hashname)
        code = code.replace('$$relation$$',relation)
        code = code.replace('$$column$$',str(column))
        return code
