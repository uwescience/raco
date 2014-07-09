#!/usr/bin/env python

""" Executes clang runner and retrieves the results """

import argparse
import os
import sys
sys.path.append('./c_test_environment')
from testquery import ClangRunner
import osutils
from osutils import Chdir

def parse_options(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('platform', metavar='P', type=str, nargs=1,
                        help='Type of platform to use: clang or grappa')

    parser.add_argument('file', help='File containing platform source program')

    ns = parser.parse_args(args)
    return ns


def main(args):
    opt = parse_options(args)
    osutils.mkdir_p("dataset")
    abspath = os.path.abspath("dataset")
    name = opt.file
    if opt.platform == 'grappa':
        #TODO
        pass
    else:
        runner = ClangRunner()
        output = runner.run(name, abspath)
    
if __name__ == "__main__":
    main(sys.argv[1:])
