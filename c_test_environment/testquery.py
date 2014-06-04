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


import abc


class PlatformRunner:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run(self, name, tmppath):
        """
        Run the query on this platform.
        Note that implementations are allowed to raise
        exceptions if execution of the query on the platform
        fails.

        @param name query name
        @return path of output file
        """
        pass


class ClangRunner(PlatformRunner):
    def __init__(self):
        pass

    def run(self, name, tmppath):
        """
        Expects the #{name}.cpp file to already exist.
        """

        envir = os.environ.copy()

        # cpp -> exe
        exe_name = './%s.exe' % (name)
        subprocess.check_call(['make', exe_name], env=envir)

        # run cpp
        testoutfn = '%s/%s.out' % (tmppath, name)
        with open(testoutfn, 'w') as outs:
            try:
                subprocess.check_call([exe_name], stdout=outs, env=envir)
            except subprocess.CalledProcessError as e1:
                # try again, this time collecting all output to print it
                try:
                    subprocess.check_call([exe_name], stderr=subprocess.STDOUT, env=envir)
                    raise e1  # just in case this doesn't fail again
                except subprocess.CalledProcessError as e2:
                    print "see executable %s" % (os.path.abspath(exe_name))
                    print subprocess.check_output(['ls', '-l', exe_name], env=envir)
                    print subprocess.check_output(['cat', '%s.cpp' % (name)], env=envir)

                    raise Exception('(Process output below)\n'+e2.output+'\n(end process output)')

        return testoutfn


from osutils import Chdir


class GrappalangRunner(PlatformRunner):
    def __init__(self):
        pass

    def run(self, name, tmppath):
        """
        Expects the #{name}.cpp file to already exist in
        $GRAPPA_HOME/applications/join.
        """

        gname = 'grappa_%s'

        envir = os.environ.copy()

        # cpp -> exe
        subprocess.check_call(['cp', '%s.cpp' % gname, envir['GRAPPA_HOME']+'/applications/join'], env=envir)
        with Chdir(envir['GRAPPA_HOME']) as grappa_dir:

          # make at base in case the cpp file is new;
          # i.e. cmake must generate the target
          with Chdir('build/Make+Release') as makedir:
            print os.getcwd()
            subprocess.check_call(['bin/distcc_make',
                                   '-j24'
                                   ], env=envir)

          with Chdir('build/Make+Release/applications/join') as appdir:
            # build the grappa application
            print os.getcwd()
            subprocess.check_call(['../../bin/distcc_make',
                                   '-j24',
                                   '%s.exe' % gname,
                                   ], env=envir)

            # run the application
            testoutfn = "%s/%s.out" % (tmppath, gname)
            with open(testoutfn, 'w') as outf:
                subprocess.check_call(['../../bin/grappa_srun',
                                       '--ppn=4',
                                       '--nnode=2',
                                       '--',
                                       '%s.exe' % gname,
                                       ],
                                        stderr=outf,
                                        stdout=outf,
                                        env=envir)

        return testoutfn


class SqliteRunner(PlatformRunner):
    """
    This platform is considered to be
    correct and provide the expected output.
    """

    def __init__(self, querypath):
        """
        @param querypath directory containing sql files
        """

        self.querypath = querypath

    def run(self, name, tmppath):
        # run sql
        querycode  = readquery("%s/%s.sql" % (self.querypath,name))
        querystr = make_query(name, querycode)

        conn = sqlite3.connect(testdbname())
        c = conn.cursor()
        expectedfn = '%s/%s.sqlite.csv' % (tmppath, name)
        with open(expectedfn, 'w') as csvfile:
            wr = csv.writer(csvfile, delimiter=' ')
            for row in c.execute(querystr):
                wr.writerow(list(row))

        return expectedfn


def checkquery(name, testplatform, trustedplatform=SqliteRunner("testqueries"), tmppath="tmp"):  # noqa
    
    """
    @param name: name of query
    @param tmppath: existing directory for temporary files
    """
 
    osutils.mkdir_p(tmppath)

    testoutfn = testplatform.run(name, tmppath)

    expectedfn = trustedplatform.run(name, tmppath)

    print "test: %s" % (name)
    verify(testoutfn, expectedfn, False)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a platform')
    parser.add_argument('queryname', metavar='q', type=str)
    parser.add_argument('--dut', dest='dut', metavar='d', type=str,
                        default="Clang", help='platform to test {Clang,Grappalang}')

    args = parser.parse_args()
    platform_runner = globals()['%sRunner' % args.dut]()

    checkquery(args.queryname, platform_runner)

