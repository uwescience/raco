#!/usr/bin/env python

""" Executes clang runner and retrieves the results """

import argparse
import os
import sys
import subprocess
sys.path.append('./c_test_environment')
from testquery import ClangRunner, GrappalangRunner
import osutils


def parse_options(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('platform', metavar='P', type=str,
                        help='Type of platform to use: clang or grappa', choices=['grappa', 'clang'])

    parser.add_argument('file', help='File containing platform source program')

    ns = parser.parse_args(args)
    return ns


def main(args):
    opt = parse_options(args)
    osutils.mkdir_p("logs")
    abspath = os.path.abspath("logs")
    name = opt.file
    if opt.platform == 'grappa':
        try:
            runner = GrappalangRunner()
            runner.run(name, abspath)
        except subprocess.CalledProcessError as e:
            print 'grappa runner for %s failed' % (name)
            print e.output
            raise
    elif opt.platform == 'clang':
        try:
            runner = ClangRunner()
            runner.run(name, abspath)
        except subprocess.CalledProcessError as e:
            print 'clang runner for %s failed' % (name)
            print e.output
            raise


if __name__ == "__main__":
    main(sys.argv[1:])
