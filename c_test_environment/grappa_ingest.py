
import re
import subprocess
import sys
import os
from c_index_strings import indexing
from convert2bin import generate_tuple_class_from_file
import csv
import argparse


p = argparse.ArgumentParser(prog=sys.argv[0])
p.add_argument("-i", dest="input_file", required=True, help="input file")
p.add_argument("-c", dest="catalog_path", help="path of catalog file, see FromFileCatalog for format", required=True)
p.add_argument("-s", dest="system", help="clang or grappa", default="clang")
args = p.parse_args(sys.argv[1:])
inputf = args.input_file
catalogfile = args.catalog_path
system = args.system


uri = 'http://sampa-gw.cs.washington.edu:1337'


def task_message(s):
    print "{0}...".format(s)

task_message("indexing")
datafile, indexfile = indexing(inputf)

# TODO: have an option to use Grappa to index the strings
# see $GRAPPA_HOME/build/Make+Release/applications/join/convert2bin.exe

task_message("generating binary converter")
cat, __convert_cpp_name = generate_tuple_class_from_file(None, catalogfile)
if len(__convert_cpp_name) > 1:
    print "WARNING: catalog had multiple entries, using the first"

#TODO: rel_key is wrong!! is public:adhoc:x need just x
rel_key, convert_cpp_name = __convert_cpp_name[0]
convert_exe_name = '{0}'.format(os.path.splitext(convert_cpp_name)[0])

task_message("building binary converter")
subprocess.check_call('make {0}'.format(convert_exe_name), shell=True)

task_message("running binary converter")
convert_stdout = subprocess.check_output('./{exe} {file} {burns} {id}'.format(exe=convert_exe_name,
                                                           file=datafile,
                                                           burns=0,
                                                           id=False), shell=True)

num_tuples = re.search("rows: (\d+)", convert_stdout).group(1)



dataset_file = '{0}.csv'.format(inputf)
scheme = cat.get_scheme(rel_key)
with open(dataset_file, 'w') as csvfile:
    w = csv.writer(csvfile, delimiter=',')
    w.writerow(['public',
                'adhoc',
                rel_key,
                uri,
                num_tuples,
                system,
                len(scheme),
                ] + scheme.get_names() + scheme.get_types())

print "data for input in " + dataset_file

# example schema for this file:
# public,adhoc,sp2bench,http://sampa-gw.cs.washington.edu:1337,922232,clang,3,subject,predicate,object,STRING_TYPE,STRING_TYPE,STRING_TYPE
