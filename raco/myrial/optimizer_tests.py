
import collections
import random

from raco.algebra import *
from raco.expression import NamedAttributeRef as AttRef
from raco.language import MyriaAlgebra
from raco.algebra import LogicalAlgebra
from raco.compile import optimize

import raco.expression as expression
import raco.scheme as scheme
import raco.myrial.myrial_test as myrial_test

class OptimizerTest(myrial_test.MyrialTestCase):

    x_scheme = scheme.Scheme([("a", "int"),("b", "int"), ("c", "int")])
    y_scheme = scheme.Scheme([("d", "int"),("e", "int"), ("f", "int")])
    x_key = "public:adhoc:X"
    y_key = "public:adhoc:Y"

    def setUp(self):
        super(OptimizerTest, self).setUp()

        random.seed(387) # make results deterministic
        rng = 20
        count = 10
        self.x_data = collections.Counter(
            [(random.randrange(rng), random.randrange(rng),
              random.randrange(rng)) for x in range(count)])
        self.y_data = collections.Counter(
            [(random.randrange(rng), random.randrange(rng),
              random.randrange(rng)) for x in range(count)])

        self.db.ingest(OptimizerTest.x_key,
                       self.x_data,
                       OptimizerTest.x_scheme)
        self.db.ingest(OptimizerTest.y_key,
                       self.y_data,
                       OptimizerTest.y_scheme)

    @staticmethod
    def logical_to_physical(lp):
        physical_plans = optimize([('root', lp)],
                                  target=MyriaAlgebra,
                                  source=LogicalAlgebra)
        return physical_plans[0][1]

    @staticmethod
    def get_count(op, claz):
        """Return the count of operator instances within an operator tree."""

        def count(_op):
            if isinstance(_op, claz):
                yield 1
            else:
                yield 0
        return sum(op.postorder(count))

    def test_push_selects(self):
        lp = StoreTemp('OUTPUT',
               Select(expression.LTEQ(AttRef("e"), AttRef("f")),
                 Select(expression.EQ(AttRef("c"),AttRef("d")),
                   Select(expression.GT(AttRef("a"),AttRef("b")),
                      CrossProduct(Scan(self.x_key, self.x_scheme),
                                   Scan(self.y_key, self.y_scheme))))))

        self.assertEquals(self.get_count(lp, Select), 3)
        self.assertEquals(self.get_count(lp, CrossProduct), 1)

        pp = self.logical_to_physical(lp)
        self.assertEquals(self.get_count(pp, Select), 1)

        # TODO: Fix these by pushing selects!
        #self.assertEquals(self.get_count(lp, CrossProduct), 0)

        self.db.evaluate(pp)
        result = self.db.get_temp_table('OUTPUT')

        expected = collections.Counter(
            [(a,b,c,d,e,f) for (a,b,c) in self.x_data
             for (d,e,f) in self.y_data if a > b and e <= f and c==d])

        self.assertEquals(result, expected)
