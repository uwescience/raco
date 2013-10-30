#!/usr/bin/env python

"""Compile a Myrial program into logical relational algebra."""

import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser
import raco.scheme
from raco import algebra
from raco import myrialang
from raco.compile import optimize
from raco.language import MyriaAlgebra

import myria

import argparse
import json
import os
import sys

def evaluate(plan, connection=None):
    if isinstance(plan, algebra.DoWhile):
        evaluate(plan.left, connection)
        evaluate(plan.right, connection)
    elif isinstance(plan, algebra.Sequence):
        for child in plan.children():
            evaluate(child, connection)
    else:
        logical = str(plan)
        physical = optimize([('', plan)], target=MyriaAlgebra, source=algebra.LogicalAlgebra)
        phys = myrialang.compile_to_json(logical, logical, physical)
        if connection is not None:
            print connection.validate_query(phys)

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
    parser.add_argument('-s', dest='server',
                        help="Hostname of the REST server", type=str, default="localhost")
    parser.add_argument('-p', dest='port',
                        help="Port of the REST server", type=int, default=8753)
    parser.add_argument('file', help='File containing Myrial source program')

    ns = parser.parse_args(args)
    return ns

class FakeCatalog(object):
    def __init__(self, catalog):
        self.catalog = catalog

    def get_scheme(self, relation_key):
        return raco.Scheme(self.catalog[relation_key])

    @classmethod
    def load_from_file(cls, path):
        with open(path) as fh:
            return cls(eval(fh.read()))

def main(args):
    opt = parse_options(args)

    # Search for a catalog definition file
    catalog_path = os.path.join(os.path.dirname(opt.file), 'catalog.py')
    catalog = None
    if os.path.exists(catalog_path):
        catalog = FakeCatalog.load_from_file(catalog_path)

    _parser = parser.Parser()
    processor = interpreter.StatementProcessor(catalog)
    myria_connection = myria.MyriaConnection(hostname=opt.server, port=opt.port)

    with open(opt.file) as fh:
        statement_list = _parser.parse(fh.read())

        processor.evaluate(statement_list)
        plan = processor.get_physical_plan()
        evaluate(plan, myria_connection)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
