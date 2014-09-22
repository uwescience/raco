import subprocess
import argparse
import sys

from raco.catalog import FromFileCatalog
from raco.language.clangcommon import StagedTupleRef

template = """
#include <tuple>
#include <string>
#include <cstring>
#include <cstdint>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <iostream>
#include "convert2bin.h"
#include "utils.h"

{definition}

int main(int argc, char** argv) {{
    if (argc < 3) {{
        std::cerr << "usage ./" << argv[0] << " [file] [burns]" << std::endl;
        exit(1);
    }}

    convert2bin_withTuple<{typ}>(argv[1], atoi(argv[2]));
}}
"""


def generate_tuple_class(rel_key, cat):
   sch = cat.get_scheme(rel_key)
   tupleref = StagedTupleRef(None, sch)
   definition = tupleref.generateDefinition()
   outfnbase = rel_key.split(':')[2]
   with open("{0}_convert.cpp".format(outfnbase), 'w') as outf:
       outf.write(template.format(definition=definition, typ=tupleref.getTupleTypename()))

   subprocess.check_output(["make", "{fn}.convert".format(fn=outfnbase)])


if __name__ == "__main__":

    p = argparse.ArgumentParser(prog=sys.argv[0])
    p.add_argument("-n", dest="name", help="name of relation [optional]. If not specified then will convert whole catalog")
    p.add_argument("-c", dest="catpath", help="path of catalog file, see FromFileCatalog for format", required=True)

    args = p.parse_args(sys.argv[1:])
   
    cat = FromFileCatalog.load_from_file(args.catpath)

    if args.name is not None:
      generate_tuple_class("public:adhoc:{0}".format(args.name), cat)
    else:
      for n in cat.get_keys():
        generate_tuple_class(n, cat)
