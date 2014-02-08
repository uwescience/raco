import random

def generate(basename, fields, tuples, datarange):
    with open(basename+str(fields), 'w') as f:
        for i in range(0,tuples):
            for j in range(0,fields):
                dat = random.randint(0, datarange)
                f.write(str(dat))
                if j<(fields-1):
                    f.write(' ')
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


if __name__ == "__main__":
    print 'generating'
    with open('importTestData.sql', 'w') as f:
        for n in ['R','S','T']:
            for nf in [1,2,3]:
                generate(n, nf, 30, 10)
                f.write(importStatement(n, nf))
