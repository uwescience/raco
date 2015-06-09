
import re
import subprocess
import sys
import os
from c_index_strings import indexing
from convert2bin import generate_tuple_class_from_file
import csv
import argparse
from raco.catalog import FromFileCatalog


p = argparse.ArgumentParser(prog=sys.argv[0])
p.add_argument("-i", dest="input_file", required=True, help="input file")
p.add_argument("-c", dest="catalog_path", help="path of catalog file, see FromFileCatalog for format", required=True)
p.add_argument("-s", dest="system", help="clang or grappa", default="clang")
p.add_argument("--external-string-index", dest="ext_index", action="store_true", help="Create string external string index that is deprecated after raco@40640adff89e1c1aade007a998b335b623ff22aa")
p.add_argument("--storage", dest="storage", default="binary", help="binary, row_ascii, row_json")
args = p.parse_args(sys.argv[1:])
inputf = args.input_file
catalogfile = args.catalog_path
system = args.system


uri = 'http://sampa-gw.cs.washington.edu:1337'


import requests
import json
class UploadConnection:
    """ TODO MERGE WITH CLANGCONNECTION
    """
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.url = 'http://{h}:{p}'.format(h=hostname, p=port)

    def _post_json(self, requrl, json_obj):
        headers = {'Content-type': 'application/json'}
        return requests.Session().post(requrl, data=json.dumps(json_obj),
                                       headers=headers)

    def upload(self, schemafile, files):
        # get upload location
        r = requests.get(self.url + '/uploadLocation')
        path = r.json()['dir']

        # copy the files to the server
        files_str = ' '.join(files + [schemafile])
        print files_str
        if self.hostname == 'localhost':
                subprocess.check_call('cp {0} {1}'.format(files_str, path),
                                      shell=True)
        else:
            subprocess.check_call('scp {0} {2}:{1}'.format(
                files_str, path, self.hostname),
                                  shell=True)

        # tell the server about the table
        schema_file_name = os.path.basename(schemafile)
        data = json.loads('{{"uploadinfo": "{0}"}}'.format(schema_file_name))
        self._post_json(self.url + '/new', data)


def task_message(s):
    print "{0}...".format(s)

upload_files = []

if args.ext_index:
    task_message("indexing")
    datafile, indexfile = indexing(inputf)
    upload_files.append(indexfile)
else:
    datafile = inputf

if args.storage == 'binary':
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

    upload_files.append(datafile+'.bin')


elif args.storage in ['row_ascii', 'row_json']:
    cat = FromFileCatalog.load_from_file(catalogfile)
    rel_key = cat.get_keys()[0]
    num_tuples = subprocess.check_output("wc -l {0} | awk '{{print $1}}'".format(inputf), shell=True)

    upload_files.append(datafile)

else:
    raise Exception("Invalid storage format {0}".format(args.storage))

schema_file = '{0}.csv'.format(inputf)
scheme = cat.get_scheme(rel_key)
with open(schema_file, 'w') as csvfile:
    w = csv.writer(csvfile, delimiter=',')
    w.writerow(['public',
                'adhoc',
                rel_key,
                uri,
                num_tuples,
                system,
                len(scheme),
                args.storage,
                ] + scheme.get_names() + scheme.get_types())

print "data for input in " + schema_file

conn = UploadConnection('localhost', '1337')
conn.upload(schema_file, upload_files)

print "successful upload of: " + schema_file + " and " + str(upload_files)


# example schema for this file:
# public,adhoc,sp2bench,http://sampa-gw.cs.washington.edu:1337,922232,clang,3,binary,subject,predicate,object,STRING_TYPE,STRING_TYPE,STRING_TYPE
