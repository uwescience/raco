# These imports are required here -- for eval inside replace_with_repr
from raco.expression import *
from raco.algebra import *
from raco.relation_key import *
from raco.scheme import *
from raco.language.myrialang import *
from raco.language.clang import *
from raco.language.grappalang import *

# NOTES: relying on import * for eval is error prone due
#        to namespace collisions
# NOTES: what to do if a operator has two constructors?


def replace_with_repr(plan):
    r = repr(plan)
    try:
        return eval(r)
    except (TypeError, AttributeError, SyntaxError):
        print 'Error with repr {r} of plan {p}'.format(r=r, p=plan)
        raise


MyriaStore(RelationKey('public', 'adhoc', 'AB'), MyriaApply(
    [('i', UnnamedAttributeRef(0, None)), ('k', UnnamedAttributeRef(1, None)),
     ('ab', UnnamedAttributeRef(2, None))],
    MyriaGroupBy([UnnamedAttributeRef(0, None), UnnamedAttributeRef(1, None)],
                 [SUM(UnnamedAttributeRef(2, None))], MyriaShuffleConsumer(
            MyriaShuffleProducer(MyriaGroupBy(
                [UnnamedAttributeRef(0, None), UnnamedAttributeRef(1, None)],
                [SUM(UnnamedAttributeRef(2, None))], MyriaApply(
                    [('i', UnnamedAttributeRef(0, None)),
                     ('k', UnnamedAttributeRef(2, None)), ('_COLUMN2_', TIMES(
                        UnnamedAttributeRef(1, None),
                        UnnamedAttributeRef(3, None)))], MyriaSymmetricHashJoin(
                        EQ(UnnamedAttributeRef(1, None),
                           UnnamedAttributeRef(3, None)), MyriaShuffleConsumer(
                            MyriaShuffleProducer(MyriaEmptyRelation(Scheme(
                                [('i', 'LONG_TYPE'), ('j', 'LONG_TYPE'),
                                 ('a', 'LONG_TYPE')])), [UnnamedAttributeRef(1,
                                                                             None)])),
                        MyriaShuffleConsumer(MyriaShuffleProducer(
                            MyriaEmptyRelation(Scheme(
                                [('j', 'LONG_TYPE'), ('k', 'LONG_TYPE'),
                                 ('b', 'LONG_TYPE')])),
                            [UnnamedAttributeRef(0, None)])),
                        [UnnamedAttributeRef(0, None),
                         UnnamedAttributeRef(2, None),
                         UnnamedAttributeRef(4, None),
                         UnnamedAttributeRef(5, None)])), []),
                                 [UnnamedAttributeRef(0, None),
                                  UnnamedAttributeRef(1, None)])), [])))

MyriaStore(RelationKey('public','adhoc','AB'),
           MyriaApply([('i', UnnamedAttributeRef(0, None)), ('k', UnnamedAttributeRef(1, None)), ('ab', UnnamedAttributeRef(2, None))],
                                       MyriaGroupBy([UnnamedAttributeRef(0, None), UnnamedAttributeRef(1, None)],
                                                    [COUNTALL()],
                                                    MyriaApply([('followee', UnnamedAttributeRef(0, None)), ('follower1', UnnamedAttributeRef(1, None))],
                                                               MyriaSymmetricHashJoin(EQ(UnnamedAttributeRef(1, None), UnnamedAttributeRef(2, None)),
                                                                                      MyriaHyperShuffleConsumer(MyriaHyperShuffleProducer(MyriaScan(RelationKey('public','adhoc','TwitterK'), Scheme([('followee', 'LONG_TYPE'), ('follower', 'LONG_TYPE')]), 10000), [UnnamedAttributeRef(0, None)], [8, 9], [0], [[0,1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16,17],[18,19,20,21,22,23,24,25,26],[27,28,29,30,31,32,33,34,35],[36,37,38,39,40,41,42,43,44],[45,46,47,48,49,50,51,52,53],[54,55,56,57,58,59,60,61,62],[63,64,65,66,67,68,69,70,71]])),
                                                                                      MyriaHyperShuffleConsumer(MyriaHyperShuffleProducer(MyriaScan(RelationKey('public','adhoc','TwitterK'), Scheme([('followee', 'LONG_TYPE'), ('follower', 'LONG_TYPE')]), 10000), [UnnamedAttributeRef(1, None)], [8, 9], [1], [[0,9,18,27,36,45,54,63],[1,10,19,28,37,46,55,64],[2,11,20,29,38,47,56,65],[3,12,21,30,39,48,57,66],[4,13,22,31,40,49,58,67],[5,14,23,32,41,50,59,68],[6,15,24,33,42,51,60,69],[7,16,25,34,43,52,61,70],[8,17,26,35,44,53,62,71]])),
                                                                                      [UnnamedAttributeRef(0, None), UnnamedAttributeRef(3, None)])),
                                           []),
                                       [UnnamedAttributeRef(0, None), UnnamedAttributeRef(1, None)]))

MyriaHyperShuffleProducer(MyriaScan(RelationKey('public','adhoc','TwitterK'), Scheme([('followee', 'LONG_TYPE'), ('follower', 'LONG_TYPE')]), 10000), [UnnamedAttributeRef(0, None)], [8, 9], [0], [[0,1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16,17],[18,19,20,21,22,23,24,25,26],[27,28,29,30,31,32,33,34,35],[36,37,38,39,40,41,42,43,44],[45,46,47,48,49,50,51,52,53],[54,55,56,57,58,59,60,61,62],[63,64,65,66,67,68,69,70,71]])
MyriaHyperShuffleProducer(MyriaScan(RelationKey('public','adhoc','TwitterK'), Scheme([('followee', 'LONG_TYPE'), ('follower', 'LONG_TYPE')]), 10000), [UnnamedAttributeRef(1, None)], [8, 9], [1], [[0,9,18,27,36,45,54,63],[1,10,19,28,37,46,55,64],[2,11,20,29,38,47,56,65],[3,12,21,30,39,48,57,66],[4,13,22,31,40,49,58,67],[5,14,23,32,41,50,59,68],[6,15,24,33,42,51,60,69],[7,16,25,34,43,52,61,70],[8,17,26,35,44,53,62,71]])

