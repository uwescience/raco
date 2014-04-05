import os
from subprocess import check_call
import sys
import sqlite3
import csv
from verifier import verify
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def readquery(fname):
    query_path = "./"
    return file(os.path.join(query_path, fname)).read()

def make_query(name, query, delim=','):

    template = """
    %(query)s
    """ 

    outputname = "%s.sqlite.out" % (name)

    query_modified = template % locals()

    return query_modified


def testquery(name, tmppath="tmp"):

    """
    @param name: name of query
    @param tmppath: existing directory for temporary files
    """
    
    mkdir_p(tmppath)
    envir = os.environ.copy()

    # cpp -> exe
    exe_name = '%s.exe' % (name)
    check_call(['make', exe_name], env=envir)
    
    # run cpp
    testoutfn = '%s/%s.out' % (tmppath, name)
    with open(testoutfn, 'w') as outs:
        check_call(['./%s' % (exe_name)], stdout=outs, env=envir)

    querycode  = readquery("%s.sql" % name)
    querystr = make_query(name, querycode)

    # run sql
    conn = sqlite3.connect('test.db')
    c = conn.cursor()
    expectedfn = '%s/%s.sqlite.csv' % (tmppath, name)
    with open(expectedfn, 'w') as csvfile:
        wr = csv.writer(csvfile, delimiter=' ')
        for row in c.execute(querystr):
            wr.writerow(list(row))
    
    print "test: %s" % (name)
    return verify(testoutfn, expectedfn, False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "usage %s <query name>" % (sys.argv[0])
        exit(1)

    name = sys.argv[1]

    testquery(name)

