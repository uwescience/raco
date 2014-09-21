import subprocess
import argparse
import sys

from raco.catalog import FromFileCatalog
from raco.language.clangcommon import StagedTupleRef

template = """
#include <tuple>
#include <string>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include "relation_io.hpp"

{definition}

int main(int argc, char** argv) {{
    if (argc < 3) {{
        std::cerr << "usage ./" << argv[0] << " [file] [burns]"
        exit(1);
    }}

    convert2bin_withTuple<{typ}>(argv[1], atoi(argv[2]));
}}
"""


def generate_tuple_class(rel_key, catalogpath):
   cat = FromFileCatalog.load_from_file(catalogpath)
   sch = cat.get_scheme(rel_key)
   tupleref = StagedTupleRef(None, sch)
   definition = tupleref.generateDefinition()
   outfnbase = rel_key.split(':')[2]
   with open("{0}.convert.cpp".format(outfnbase), 'w') as outf:
       outf.write(template.format(definition=definition, typ=tupleref.getTupleTypename()))

   subprocess.check_output("make {fn}.convert".format(fn=outfnbase))


if __name__ == "__main__":

    p = argparse.ArgumentParser(prog=sys.argv[0])
    p.add_argument("-n", dest="name", help="name of relation", required=True)
    p.add_argument("-c", dest="catpath", help="path of catalog file, see FromFileCatalog for format", required=True)

    args = p.parse_args(sys.argv[1:])

    generate_tuple_class("public:adhoc:{0}".format(args.name), args.catpath)
