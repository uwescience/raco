import os
import subprocess
import sys
import sqlite3
import csv
from verifier import verify, verify_store
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
        try:
            subprocess.check_output(['make', 'clean'],
                                    stderr=subprocess.STDOUT,
                                    env=envir)
            subprocess.check_output(['make', exe_name],
                                    stderr=subprocess.STDOUT,
                                    env=envir)
        except subprocess.CalledProcessError as e:
            print 'make {exe} failed:'.format(exe=exe_name)
            print e.output
            raise

        # run cpp
        testoutfn = '%s/%s.out' % (tmppath, name)
        try:
            with open(testoutfn, 'w') as outs:
                subprocess.check_call([exe_name], stdout=outs, env=envir)
        except subprocess.CalledProcessError:
            print "see executable %s" % (os.path.abspath(exe_name))
            print subprocess.check_output(['ls', '-l', exe_name], env=envir)
            print subprocess.check_output(['cat', testoutfn], env=envir)
            raise

        return testoutfn


from osutils import Chdir


class GrappalangRunner(PlatformRunner):
    def __init__(self, binary_input=True):
      self.binary_input = binary_input

    def run(self, name, tmppath):
        """
        Expects the #{name}.cpp file to already exist in
        $GRAPPA_HOME/applications/join.
        """

        gname = 'grappa_%s' % name

        envir = os.environ.copy()

        # cpp -> exe

        # call configure only if a previous version does not exist
        # (i.e., the cmake target likely does not exist yet)
        need_configure = not os.path.isfile(
            os.path.join(
                envir['GRAPPA_HOME'],
                'build/Make+Release/applications/join',
                "{0}.exe".format(gname)))

        subprocess.check_call(['cp', '%s.cpp' % gname, envir['GRAPPA_HOME']+'/applications/join'], env=envir)

        if need_configure:
            subprocess.check_call(['./grappa_detect_new_files.sh'], env=envir)

        with Chdir(envir['GRAPPA_HOME']) as grappa_dir:
          # make at base in case the cpp file is new;
          # i.e. cmake must generate the target
          with Chdir('build/Make+Release') as makedir:
            print os.getcwd()
            #subprocess.check_call(['bin/distcc_make', '-j'], env=envir)
            subprocess.check_call(['make', '-j', '8'], env=envir)

            """subprocess.check_call(['bin/distcc_make',
                                   '-j24'
                                   ], env=envir)
                                   """

          with Chdir('build/Make+Release/applications/join') as appdir:
            # build the grappa application
            print os.getcwd()
            print subprocess.check_call(['make', '-j', '8', '%s.exe' % gname], env=envir)

            # subprocess.check_call(['../../bin/distcc_make',
            #                        '-j24',
            #                        '%s.exe' % gname,
            #                        ], env=envir)


            # run the application
            testoutfn = "%s/%s.out" % (tmppath, gname)
            with open(testoutfn, 'w') as outf:
                subprocess.check_call(['../../bin/grappa_srun',
                                       '--ppn=8',
                                       '--nnode=4',
                                       '--',
                                       '%s.exe' % gname,
                                       '--bin={0}'.format(self.binary_input),
                                       '--vmodule=%s=2' % gname  # result out
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
    abstmppath = os.path.abspath(tmppath)

    testoutfn = testplatform.run(name, abstmppath)

    expectedfn = trustedplatform.run(name, abstmppath)

    print "test: %s" % (name)
    verify(testoutfn, expectedfn, False)


def checkstore(name, testplatform, trustedplatform=SqliteRunner("testqueries"), tmppath="tmp"):  # noqa

    """
    @param name: name of query
    @param tmppath: existing directory for temporary files
    """

    osutils.mkdir_p(tmppath)
    abstmppath = os.path.abspath(tmppath)
    testplatform.run(name, abstmppath)
    trustedplatform.run(name, abstmppath)
    testoutfn = name
    expectedfn = "%s/%s.sqlite.csv" %(abstmppath, name)

    print "test: %s" % (name)
    verify_store(testoutfn, expectedfn, False)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a platform')
    parser.add_argument('queryname', metavar='q', type=str)
    parser.add_argument('--dut', dest='dut', metavar='d', type=str,
                        default="Clang", help='platform to test {Clang,Grappalang}')

    args = parser.parse_args()
    platform_runner = globals()['%sRunner' % args.dut]()

    checkquery(args.queryname, platform_runner)

