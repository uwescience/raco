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
import time

def evaluate(plan, connection=None, validate=False):
    if isinstance(plan, algebra.DoWhile):
        # Left is the body of the loop
        evaluate(plan.left, connection, validate)
        # Right is just a scan of the "continue" relation. Don't execute it,
        # just use it to get the name of that relation.

        if not connection or validate:
            return
        if isinstance(plan.right, algebra.ScanTemp):
            name = plan.right.name
        elif isinstance(plan.right, algebra.Scan):
            name = plan.right.relation_key
        else:
            print >> sys.stderr, "Unknown while condition %s of class %s. executing then quitting loop." % (plan.right, plan.right.__class__)
            evaluate(plan.right, connection, validate)
            return
        user_name, program_name, relation_name = myrialang.resolve_relation_key(name)
        relation_key = {'userName' : user_name,
                        'programName' : program_name,
                        'relationName' : relation_name }
        d = connection.download_dataset(relation_key)
        if d[0].values()[0]:
            evaluate(plan, connection, validate)


    elif isinstance(plan, algebra.Sequence):
        for child in plan.children():
            evaluate(child, connection, validate)
    else:
        logical = str(plan)
        physical = plan
        phys = myrialang.compile_to_json(logical, logical, physical)
        if connection is not None:
            if validate:
                print json.dumps(connection.validate_query(phys))
            else:
                print >> sys.stderr, "Submitting %s" % logical
                query = connection.submit_query(phys)
                while query['status'] in [ 'ACCEPTED', 'RUNNING' , 'PAUSED' ]:
                    time.sleep(0.0001)
                    query = connection.get_query_status(query['queryId'])
                if query['status'] != 'SUCCESS':
                    raise IOError('Query %s failed: %s' % (logical, json.dumps(query)))
                else:
                    print >> sys.stderr, 'Query %s finished in %d ms' % (logical, query['elapsed_nanos'] / 1e6)
        print

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
    parser.add_argument('-v', dest='validate', action="store_true",
                        help="Validate the program, but do not submit it")
    parser.add_argument('file', help='File containing Myrial source program')

    ns = parser.parse_args(args)
    return ns

class FakeCatalog(object):
    def __init__(self, catalog):
        self.catalog = catalog

    def get_scheme(self, relation_key):
        string_key = str(relation_key)
        return raco.Scheme(self.catalog[string_key])

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

    # For sigma clipping, we need to ingest the points file
    try:
        myria_connection.upload_fp(
                { 'userName' : 'public', 'programName' : 'adhoc', 'relationName':'sc_points'},
                { 'columnNames' : ['v'], 'columnTypes' : ['DOUBLE_TYPE'] },
                open('examples/sigma_clipping_points.txt', 'r'))
    except myria.MyriaError as e:
        if '409' in str(e):
            # Dataset has already been ingested, we can safely ignore
            pass
        else:
            raise e

    with open(opt.file) as fh:
        statement_list = _parser.parse(fh.read())

        processor.evaluate(statement_list)
        plan = processor.get_physical_plan()
        evaluate(plan, myria_connection, opt.validate)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
