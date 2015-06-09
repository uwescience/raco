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
    parser.add_argument('--query', help='File containing myrial query')
    parser.add_argument('--catalog', help='File containing catalog')

    ns = parser.parse_args(args)
    return ns


from raco.language.clang import CCAlgebra
from raco.language.grappalang import GrappaAlgebra
from raco.catalog import FromFileCatalog
from raco.language.clangcommon import EMIT_FILE
from clang_processor import ClangProcessor


def main(args):
    opt = parse_options(args)
    osutils.mkdir_p("logs")
    abspath = os.path.abspath("logs")
    name = opt.file

    if opt.query:
        if opt.catalog is None:
            raise Exception("--query also requires a --catalog")

        with open(opt.query, 'r') as f:
            qt = f.read()

        target_alg = CCAlgebra(emit_print=EMIT_FILE)
        if opt.platform == 'grappa':
            target_alg = GrappaAlgebra(emit_print=EMIT_FILE)
        ClangProcessor(FromFileCatalog.load_from_file(opt.catalog))\
            .write_source_code(qt, name, target_alg=target_alg)

    if opt.platform == 'grappa':
        runner = GrappalangRunner()
        runner.run(name, abspath)
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
