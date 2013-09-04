#!/usr/bin/env python

import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser

import argparse
import sys

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
    processor = interpreter.StatementProcessor(sys.stdout)

    with open(opt.file) as fh:
        statement_list = _parser.parse(fh.read())

        if opt.parse_only:
            print statement_list
        else:
            processor.evaluate(statement_list)
