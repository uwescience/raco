#!/usr/bin/env python

import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
from raco import algebra

import argparse
import sys

def print_pretty_plan(plan, indent=0):
    if isinstance(plan, algebra.DoWhile):
        print '%s%s' % (' ' * indent, plan.shortStr())
        print_pretty_plan(plan.left, indent + 4)
        print_pretty_plan(plan.right, indent + 4)
    elif isinstance(plan, algebra.Sequence):
        print '%s%s' % (' ' * indent, plan.shortStr())
        for child in plan.children():
            print_pretty_plan(child, indent + 4)
    else:
        print '%s%s' % (' ' * indent, plan)

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
            plan = processor.get_output()
            print_pretty_plan(plan)
