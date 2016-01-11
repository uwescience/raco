
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
p.add_argument("-n", dest="relation_name", required=True, help="name of relation")
p.add_argument("-s", dest="system", help="cpp or grappa", default="cpp")
p.add_argument("--splits", dest="splits", action="store_true", help="input file is base directory of file splits (e.g. hdfs)")
p.add_argument("--softlink-data", dest="softlink_data", action="store_true", help="data file softlinked rather than copied")
p.add_argument("--local-softlink-data", dest="local_softlink_data", action="store_true", help="softlink locally, only use if --host!=localhost but want local softlink (e.g. NFS)")
p.add_argument("--allow-failed-upload", dest="allow_failed_upload", action="store_true", help="if softlinking on then still softlink even if uploading data fails")
p.add_argument("--host", dest="host", help="hostname of server", default="localhost")
p.add_argument("--port", dest="port", help="port server is listening on", default=1337)
p.add_argument("--external-string-index", dest="ext_index", action="store_true", help="Create string external string index that is deprecated after raco@40640adff89e1c1aade007a998b335b623ff22aa")
p.add_argument("--storage", dest="storage", default="binary", help="binary, row_ascii, row_json")
p.add_argument("--delim", dest="delim", default=' ', help="delimiter for ingesting ascii csv data")
args = p.parse_args(sys.argv[1:])
inputf = args.input_file
catalogfile = args.catalog_path
system = args.system


uri = 'http://{h}:{p}'.format(h=args.host, p=args.port)


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

    def _get_upload_path(self):
        # get upload location
        r = requests.get(self.url + '/uploadLocation')
        path = r.json()['dir']
        return path

    def softlink(self, files):
        if len(files) == 0:
            return

        assert self.hostname == 'localhost' or args.local_softlink_data, "softlink currently supported " \
                                "only for localhost unless using --local-softlink-data"

        path = self._get_upload_path()

        for f in files:
          subprocess.check_call('ln -s {target} {name}'.format(
                target=f, name=path+'/'+os.path.basename(f)), shell=True)

    def upload(self, schemafile, files):
        path = self._get_upload_path()

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
link_files = []
def add_data_file(f):
    if args.softlink_data:
        link_files.append(f)
    else:
        upload_files.append(f)

if args.ext_index:
    assert not args.splits, "--splits and --external-string-indexing not currently compatible"
    task_message("indexing")
    datafile, indexfile = indexing(inputf)
    upload_files.append(indexfile)
else:
    datafile = inputf

if args.storage == 'binary':
    assert not args.splits, "--splits and --storage=binary not currently compatible"

    # TODO: have an option to use Grappa to index the strings
    # see $GRAPPA_HOME/build/Make+Release/applications/join/convert2bin.exe

    task_message("generating binary converter")
    cat, rel_key, convert_cpp_name = generate_tuple_class_from_file(
        args.relation_name,
        catalogfile)

    #TODO: rel_key is wrong!! is public:adhoc:x need just x
    convert_exe_name = '{0}'.format(os.path.splitext(convert_cpp_name)[0])

    task_message("building binary converter")
    subprocess.check_call('make {0}'.format(convert_exe_name), shell=True)

    task_message("running binary converter")
    convert_stdout = subprocess.check_output('./{exe} {file} "{delim}" {burns} {id}'.format(exe=convert_exe_name,
                                                               file=datafile,
                                                               delim=args.delim,
                                                               burns=0,
                                                               id=False), shell=True)

    num_tuples = re.search("rows: (\d+)", convert_stdout).group(1)

    add_data_file(datafile+'.bin')


elif args.storage in ['row_ascii', 'row_json']:
    cat = FromFileCatalog.load_from_file(catalogfile)
    rel_key = cat.get_keys()[0]

    if args.splits:
        num_tuples = subprocess.check_output("wc -l {0}/part-* "
                                             "| tail -n 1 "
                                             "| awk '{{print $1}}'".format(inputf)
                                             , shell=True)
    else:
        num_tuples = subprocess.check_output("wc -l {0} | awk '{{print $1}}'".format(inputf), shell=True)

    add_data_file(datafile)

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

conn = UploadConnection(args.host, args.port)

try:
    conn.upload(schema_file, upload_files)
except Exception as e:
    if not args.allow_failed_upload:
        raise e

conn.softlink(link_files)

print "successful upload of: " + schema_file + " and " + str(upload_files)


# example schema for this file:
# public,adhoc,sp2bench,http://sampa-gw.cs.washington.edu:1337,922232,cpp,3,binary,subject,predicate,object,STRING_TYPE,STRING_TYPE,STRING_TYPE
