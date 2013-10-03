#!/usr/bin/env python

import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
from raco import algebra

import argparse
import sys

def print_pretty_plan(plan):
    for (label, root_op) in plan:
        if isinstance(root_op, algebra.DoWhile):
            print root_op.shortStr()
            for inner_statement in root_op.body_ops:
                print '\t%s' % inner_statement
            print '\tContinue if: %s' % root_op.term_op
        else:
            print "%s = %s" % (label, root_op)

def parse_options(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='parse_only',
                        help="Parse only", action='store_true')
    parser.add_argument('file', help='File containing Myrial source program')

    ns = parser.parse_args(args)
    return ns

if __name__ == "__main__":

    opt = parse_options(sys.argv[1:])
    _parser = parser.Parser()
    processor = interpreter.StatementProcessor()

    with open(opt.file) as fh:
        statement_list = _parser.parse(fh.read())

        if opt.parse_only:
            print statement_list
        else:
            processor.evaluate(statement_list)
            print_pretty_plan(processor.output_symbols)
