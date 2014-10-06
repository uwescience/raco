import random
import os
from subprocess import check_call

def get_name(basename, fields):
    return basename+str(fields)

def generate_random(basename, fields, tuples, datarange):
    fn = get_name(basename, fields)
    with open(fn, 'w') as f:
        print "generating %s" % (os.path.abspath(fn))
        for i in range(0,tuples):
            for j in range(0,fields):
                dat = random.randint(0, datarange)
                f.write(str(dat))
                if j<(fields-1):
                    f.write(' ')
            f.write("\n")

def generate_last_sequential(basename, fields, tuples, datarange):
    fn = get_name(basename, fields)
    with open(fn, 'w') as f:
        print "generating %s" % (os.path.abspath(fn))
        for i in range(0,tuples):
            for j in range(0,fields-1):
                dat = random.randint(0, datarange)
                f.write(str(dat))
                if j<(fields-1):
                    f.write(' ')
            f.write(str(i))  # sequential attribute

            f.write("\n")

def importStatement(basename, fields):
    template = """create table %(basename)s%(fields)s(%(fielddcls)s);
.separator " "
.import %(basename)s%(fields)s %(basename)s%(fields)s
    
"""

    fielddcls = ""
    names = ['a','b','c']
    for i in range(0,fields):
        fielddcls += names[i] + ' integer'
        if i < fields-1:
            fielddcls +=", "

    text = template % locals()
    return text

def need_generate(cpdir=None):
  if cpdir:
    return not os.path.isfile(os.path.join(cpdir, 'test.db'))
  else:
    return not os.path.isfile('test.db')

def generate_default(cpdir=None):
    print 'generating'
    with open('importTestData.sql', 'w') as f:
        for n, genfunc in [('R', generate_random),
                           ('S', generate_random),
                           ('T', generate_random),
                           ('I', generate_last_sequential)]:
            for nf in [1, 2, 3]:
                genfunc(n, nf, 30, 10)
                f.write(importStatement(n, nf))
                if cpdir:
                    fn = get_name(n, nf)
                    check_call(['ln', '-fs', os.path.abspath(fn), cpdir])

        if cpdir:
          check_call(['ln', '-fs', os.path.abspath('test.db'), cpdir])

    # import to sqlite3
    print 'importing'
    check_call(['rm', '-f', 'test.db'])
    with open('importTestData.sql', 'r') as f:
       check_call(['sqlite3', 'test.db'], stdin=f) 


if __name__ == "__main__":
    generate_default()

