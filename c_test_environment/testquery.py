import os
import subprocess
import sys
import sqlite3
import csv
from verifier import verify
import osutils

def testdbname():
    return 'test.db'

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


def checkquery(name, tmppath="tmp", querypath="testqueries"):
    
    """
    @param name: name of query
    @param tmppath: existing directory for temporary files
    """
 
    osutils.mkdir_p(tmppath)
    envir = os.environ.copy()

    # cpp -> exe
    exe_name = './%s.exe' % (name)
    subprocess.check_call(['make', exe_name], env=envir)
    
    # run cpp
    testoutfn = '%s/%s.out' % (tmppath, name)
    with open(testoutfn, 'w') as outs:
        try:
            subprocess.check_call(['%s' % (exe_name)], stdout=outs, env=envir)
        except subprocess.CalledProcessError as e1:
            # try again, this time collecting all output to print it
            try:
                subprocess.check_output([exe_name], stderr=subprocess.STDOUT, env=envir)
                raise e1  # just in case this doesn't fail again
            except subprocess.CalledProcessError as e2:
                print "see executable %s" % (exe_name)
                print subprocess.check_output(['ls', '-l', exe_name], env=envir)
                print subprocess.check_output(['ls', '-l', '{R,S,T}{1,2,3}'], env=envir)
                print subprocess.check_output(['cat', exe_name], env=envir)
                 
                raise Exception('(Process output below)\n'+e2.output+'\n(end process output)')

    querycode  = readquery("%s/%s.sql" % (querypath,name))
    querystr = make_query(name, querycode)

    # run sql
    conn = sqlite3.connect(testdbname())
    c = conn.cursor()
    expectedfn = '%s/%s.sqlite.csv' % (tmppath, name)
    with open(expectedfn, 'w') as csvfile:
        wr = csv.writer(csvfile, delimiter=' ')
        for row in c.execute(querystr):
            wr.writerow(list(row))
    
    print "test: %s" % (name)
    verify(testoutfn, expectedfn, False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "usage %s <query name>" % (sys.argv[0])
        exit(1)

    name = sys.argv[1]

    checkquery(name)

