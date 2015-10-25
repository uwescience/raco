import subprocess
import argparse
import sys

from raco.catalog import FromFileCatalog
from raco.backends.cpp.cppcommon import StagedTupleRef

"""
given a schema, creates a C++ program to convert csv data to a binary format
"""

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
#include "radish_utils.h"
#include "strings.h"

{definition}

int main(int argc, char * const argv[]) {{
    if (argc < 4) {{
        std::cerr << "usage: " << argv[0] << " [file] [delim char] [burns] [add_id?]" << std::endl;
        exit(1);
    }}

    convert2bin_withTuple<{typ}>(argv[1], argv[2][0], atoi(argv[3]), atoi(argv[4]));
}}
"""


def generate_tuple_class(rel_key, cat):
   sch = cat.get_scheme(rel_key)
   tupleref = StagedTupleRef(None, sch)
   definition = tupleref.generateDefinition()
   outfnbase = rel_key.split(':')[2]
   cpp_name = "{0}.convert.cpp".format(outfnbase)
   with open(cpp_name, 'w') as outf:
       outf.write(template.format(definition=definition, typ=tupleref.getTupleTypename()))

   subprocess.check_output(["make", "{fn}.convert".format(fn=outfnbase)])
   return cpp_name


def generate_tuple_class_from_file(name, catpath):
    cat = FromFileCatalog.load_from_file(catpath)

    if name is not None:
        rel_key = "public:adhoc:{0}".format(name)
        return cat, rel_key, generate_tuple_class(rel_key, cat)
    else:
        return cat, [(n, generate_tuple_class(n, cat)) for n in cat.get_keys()]


if __name__ == "__main__":

    p = argparse.ArgumentParser(prog=sys.argv[0])
    p.add_argument("-n", dest="name", help="name of relation [optional]. If not specified then will convert whole catalog")
    p.add_argument("-c", dest="catpath", help="path of catalog file, see FromFileCatalog for format", required=True)

    args = p.parse_args(sys.argv[1:])
    generate_tuple_class_from_file(args.name, args.catpath)
   
