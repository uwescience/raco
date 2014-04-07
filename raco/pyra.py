from raco.scheme import Scheme


class Tuple(object):
    def __init__(self, tup, sch):
        for (n, t), v in zip(sch, tup):
            self.__dict__[n] = t(v)


class Relation(object):
    def __init__(self, scheme, tuples=None, name=None):
        if tuples is None:
            tuples = []
        self.scheme = scheme
        self.tuples = []
        self.name = name
        for t in tuples:
            self.insert(t)

    def getScheme(self):
        return self.scheme

    def __getitem__(self, i):
        return self.tuples.__getitem__(i)

    def __setitem__(self, k, tup):
        self.scheme.typecheck(tup)
        self.tuples.__setitem__(k, tup)

    def insert(self, tup):
        self.scheme.typecheck(tup)
        self.tuples.append(tup)

    def iterdicts(self):
        for t in self.tuples:
            tt = Tuple(t, sch)
            yield tt

    def __iter__(self):
        return self.tuples.__iter__()

    def __len__(self):
        return len(self.tuples)

    def __str__(self):
        return "%s\n%s" % (self.scheme,
                           "\n".join([str(t) for t in self.tuples]))


def product(rs, ss):
    for r in rs:
        for s in ss:
            yield r + s


def scan(rname, db):
    return db[rname]


def select(condition, R):
    sch = R.getScheme()
    result = Relation(sch)
    for t in R:
        if condition(Tuple(t, sch)):
            result.insert(t)
    return result


def project(attributes, R):
    sch = R.getScheme()
    result = Relation(sch.subScheme(attributes))
    ext = sch.getExtractor(attributes)
    for t in R:
        result.insert(ext(t))


def hash(attributes, R):
    sch = R.getScheme()
    ext = sch.getExtractor(attributes)
    d = {}
    for t in R:
        d.setdefault(ext(t), []).append(t)
    return d


def dump(R):
    print R

# def probe(attributes, hR, S, reduce=product):
#   ext = S.getScheme().getExtractor(attributes)
#   d = {}
#   for st in S:
#     rt = hR[ext(st)]
#     d[ext(st)] = rt
#     result.insert(rt)


def hashjoin(attributes, Left, Right):
    keys = zip(*attributes)
    ht = hash(keys[0], Left)
    # probe(attributes, ht, Right)
    result = Relation(Left.getScheme() + Right.getScheme())
    ext = Right.getScheme().getExtractor(keys[1])
    for t in Right:
        k = ext(t)
        if k in ht:
            lefts = ht[k]
            for lt in lefts:
                result.insert(lt + t)
    return result

if __name__ == '__main__':
    sch = Scheme([("subject", int), ("predicate", int), ("object", int)])
    rel = Relation(sch, [(1, 2, 2), (1, 3, 2), (2, 2, 1), (2, 3, 3)])
    sub = sch.subScheme(["subject"])
    print rel
    print sub
    print sch + sub
    sr = select(lambda t: t.predicate == 3, rel)
    print sr
    print hashjoin([("object", "subject")], sr, rel)

"""
Example program:

import pyra
import sampledb

R = pyra.scan("R", sampledb.__dict__)

sR = pyra.select(lambda t: t.predicate == 'knows', R)

sS = pyra.select(lambda t: t.predicate == 'holdsAccount', R)

sT = pyra.select(lambda t: t.predicate == 'accountServiceHomepage', R)

sRsS = pyra.hashjoin([("object","subject")], sR, sS)

sRsSsT = pyra.hashjoin([("object1", "subject")], sRsS, sT)

pyra.dump(sRsSsT)

"""
