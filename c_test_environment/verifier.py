import re
import sys

"""
Checks two query outputs for equality
"""

class ComparableFloat(object):
    """Wrapper to use == on floats safely"""

    def __init__(self, d, ep=0.00001):
        self._raw = d
        self._ep = ep

    def __eq__(self, other):
        print "comparing ", self, other
        if type(other) is ComparableFloat:
            return abs(other._raw - self._raw) <= self._ep
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "{0}({1})".format(ComparableFloat.__name__, self._raw)
    
    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self._raw)

doublepat = re.compile(r'^-?\d+[.]\d+$')
intpat = re.compile(r'^-?\d+$')
def parse_value(value):
    if doublepat.match(value):
        n = float(value)
        return ComparableFloat(n)
    elif intpat.match(value):
        return int(value)
    else:
        return value


def verify(testout, expected, ordered):
    tuplepat = re.compile(r'Materialized\(([-\dA-Za-z,. ]+)\)')
    test = ({}, [])
    expect = ({}, [])

    def addTuple(tc, t):
        if ordered:
            tcl = tc[1]
            tcl.append(t)
        else:
            tcs = tc[0]
            if t not in tcs:
                tcs[t] = 1
            else:
                tcs[t]+=1

    with open(testout, 'r') as file:
        for line in file.readlines():
            m = tuplepat.search(line)
            if m:
                tlist = []
                for number in m.group(1).split(','):
                    if number=='':
                        # last one
                        break
                    tlist.append(parse_value(number))

                t = tuple(tlist)
                addTuple(test, t)

    with open(expected, 'r') as file:
        for line in file.readlines():
            tlist = []
            for number in line.split():
                tlist.append(parse_value(number))

            t = tuple(tlist)
            addTuple(expect, t)

    print test
    print expect
    assert test == expect, "\n test:  %s !=\n expect:%s" % (test, expect)
    print "pass"


def verify_store(testout, expected, ordered):
    numpat = re.compile(r'(\d+)')
    test = ({}, [])
    expect = ({}, [])

    def addTuple(tc, t):
        if ordered:
            tcl = tc[1]
            tcl.append(t)
        else:
            tcs = tc[0]
            if t not in tcs:
                tcs[t] = 1
            else:
                tcs[t]+=1

    with open(testout, 'r') as file:
        for line in file.readlines():
            tlist = []
            for number in numpat.finditer(line):
                tlist.append(int(number.group(0)))
                
            t = tuple(tlist)
            addTuple(test, t)

    with open(expected, 'r') as file:
        for line in file.readlines():
            tlist = []
            for number in numpat.finditer(line):
                tlist.append(int(number.group(0)))

            t = tuple(tlist)
            addTuple(expect, t)

    print test
    print expect
    assert test == expect, "\n test:  %s !=\n expect:%s" % (test, expect)
    print "pass"


if __name__ == '__main__':
    testout=sys.argv[1]
    expected=sys.argv[2]

    ordered = False
    if len(sys.argv) > 3:
        if sys.argv[3] == 'o':
            ordered = True 

    verify(testout, expected, ordered)
 
