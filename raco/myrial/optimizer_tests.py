import collections
import random
import sys
import re

from raco.algebra import *
from raco.expression import NamedAttributeRef as AttRef
from raco.expression import UnnamedAttributeRef as AttIndex
from raco.expression import StateVar
from raco.expression import aggregate

from raco.backends.myria import (
    MyriaShuffleConsumer, MyriaShuffleProducer, MyriaHyperCubeShuffleProducer,
    MyriaBroadcastConsumer, MyriaQueryScan, MyriaSplitConsumer, MyriaUnionAll,
    MyriaBroadcastProducer, MyriaScan, MyriaSelect, MyriaSplitProducer,
    MyriaDupElim, MyriaGroupBy, MyriaIDBController, MyriaSymmetricHashJoin,
    compile_to_json)
from raco.backends.myria import (MyriaLeftDeepTreeAlgebra,
                                 MyriaHyperCubeAlgebra)
from raco.compile import optimize
from raco import relation_key
from raco.catalog import FakeCatalog

import raco.scheme as scheme
import raco.myrial.myrial_test as myrial_test
from raco import types


class OptimizerTest(myrial_test.MyrialTestCase):

    x_scheme = scheme.Scheme([("a", types.LONG_TYPE), ("b", types.LONG_TYPE), ("c", types.LONG_TYPE)])  # noqa
    y_scheme = scheme.Scheme([("d", types.LONG_TYPE), ("e", types.LONG_TYPE), ("f", types.LONG_TYPE)])  # noqa
    z_scheme = scheme.Scheme([('src', types.LONG_TYPE), ('dst', types.LONG_TYPE)])  # noqa
    part_scheme = scheme.Scheme([("g", types.LONG_TYPE), ("h", types.LONG_TYPE), ("i", types.LONG_TYPE)])  # noqa
    broad_scheme = scheme.Scheme([("j", types.LONG_TYPE), ("k", types.LONG_TYPE), ("l", types.LONG_TYPE)])  # noqa
    x_key = relation_key.RelationKey.from_string("public:adhoc:X")
    y_key = relation_key.RelationKey.from_string("public:adhoc:Y")
    z_key = relation_key.RelationKey.from_string("public:adhoc:Z")
    part_key = relation_key.RelationKey.from_string("public:adhoc:part")
    broad_key = relation_key.RelationKey.from_string("public:adhoc:broad")
    part_partition = RepresentationProperties(
        hash_partitioned=frozenset([AttIndex(1)]))
    broad_partition = RepresentationProperties(broadcasted=True)
    random.seed(387)  # make results deterministic
    rng = 20
    count = 30
    z_data = collections.Counter([(1, 2), (2, 3), (1, 2), (3, 4)])
    x_data = collections.Counter(
        [(random.randrange(rng), random.randrange(rng),
          random.randrange(rng)) for _ in range(count)])
    y_data = collections.Counter(
        [(random.randrange(rng), random.randrange(rng),
          random.randrange(rng)) for _ in range(count)])
    part_data = collections.Counter(
        [(random.randrange(rng), random.randrange(rng),
          random.randrange(rng)) for _ in range(count)])
    broad_data = collections.Counter(
        [(random.randrange(rng), random.randrange(rng),
          random.randrange(rng)) for _ in range(count)])

    def setUp(self):
        super(OptimizerTest, self).setUp()
        self.db.ingest(self.x_key, self.x_data, self.x_scheme)
        self.db.ingest(self.y_key, self.y_data, self.y_scheme)
        self.db.ingest(self.z_key, self.z_data, self.z_scheme)
        self.db.ingest(self.part_key, self.part_data, self.part_scheme,
                       self.part_partition)  # "partitioned" table
        self.db.ingest(self.broad_key, self.broad_data,
                       self.broad_scheme, self.broad_partition)

    @staticmethod
    def logical_to_physical(lp, **kwargs):
        if kwargs.get('hypercube', False):
            algebra = MyriaHyperCubeAlgebra(FakeCatalog(64))
        else:
            algebra = MyriaLeftDeepTreeAlgebra()
        return optimize(lp, algebra, **kwargs)

    @staticmethod
    def get_count(op, claz):
        """Return the count of operator instances within an operator tree."""

        def count(_op):
            if isinstance(_op, claz):
                yield 1
            else:
                yield 0
        return sum(op.postorder(count))

    @staticmethod
    def get_num_select_conjuncs(op):
        """Get the number of conjunctions within all select operations."""
        def count(_op):
            if isinstance(_op, Select):
                yield len(expression.extract_conjuncs(_op.condition))
            else:
                yield 0
        return sum(op.postorder(count))

    def test_push_selects(self):
        """Test pushing selections into and across cross-products."""
        lp = StoreTemp('OUTPUT',
               Select(expression.LTEQ(AttRef("e"), AttRef("f")),
                 Select(expression.EQ(AttRef("c"), AttRef("d")),
                   Select(expression.GT(AttRef("a"), AttRef("b")),
                      CrossProduct(Scan(self.x_key, self.x_scheme),
                                   Scan(self.y_key, self.y_scheme))))))  # noqa

        self.assertEquals(self.get_count(lp, Select), 3)
        self.assertEquals(self.get_count(lp, CrossProduct), 1)

        pp = self.logical_to_physical(lp)
        self.assertIsInstance(pp.input, MyriaSplitConsumer)
        self.assertIsInstance(pp.input.input.input, Join)
        self.assertEquals(self.get_count(pp, Select), 2)
        self.assertEquals(self.get_count(pp, CrossProduct), 0)

        self.db.evaluate(pp)
        result = self.db.get_temp_table('OUTPUT')
        expected = collections.Counter(
            [(a, b, c, d, e, f) for (a, b, c) in self.x_data
             for (d, e, f) in self.y_data if a > b and e <= f and c == d])
        self.assertEquals(result, expected)

    def test_collapse_applies(self):
        """Test pushing applies together."""
        lp = StoreTemp('OUTPUT',
               Apply([(None, AttIndex(1)), ('w', expression.PLUS(AttIndex(0), AttIndex(0)))],       # noqa
                 Apply([(None, AttIndex(1)), (None, AttIndex(0)), (None, AttIndex(1))],             # noqa
                   Apply([('x', AttIndex(0)), ('y', expression.PLUS(AttIndex(1), AttIndex(0)))],    # noqa
                     Apply([(None, AttIndex(0)), (None, AttIndex(1))],
                           Scan(self.x_key, self.x_scheme))))))  # noqa

        self.assertEquals(self.get_count(lp, Apply), 4)

        pp = self.logical_to_physical(lp)
        self.assertIsInstance(pp.input, Apply)
        self.assertEquals(self.get_count(pp, Apply), 1)

        expected = collections.Counter(
            [(b, a + a) for (a, b, c) in
             [(b, a, b) for (a, b) in
              [(a, b + a) for (a, b) in
                [(a, b) for (a, b, c) in self.x_data]]]])  # noqa
        self.db.evaluate(pp)
        result = self.db.get_temp_table('OUTPUT')
        self.assertEquals(result, expected)

    def test_select_count_star(self):
        """Test that we don't generate 0-length applies from a COUNT(*)."""
        lp = StoreTemp('OUTPUT',
                       GroupBy([], [expression.COUNTALL()],
                               Scan(self.x_key, self.x_scheme)))

        self.assertEquals(self.get_count(lp, GroupBy), 1)

        pp = self.logical_to_physical(lp)
        self.assertIsInstance(pp.input.input.input, GroupBy)
        # SplitC.SplitP.GroupBy.CollectP.CollectC.GroupBy.Apply
        apply = pp.input.input.input.input.input.input.input
        self.assertIsInstance(apply, Apply)
        self.assertEquals(self.get_count(pp, Apply), 1)
        self.assertEquals(len(apply.scheme()), 1)

        expected = collections.Counter([(len(self.x_data),)])
        self.db.evaluate(pp)
        result = self.db.get_temp_table('OUTPUT')
        self.assertEquals(result, expected)

    def test_projects_apply_join(self):
        """Test column selection both Apply into ProjectingJoin
        and ProjectingJoin into its input.
        """
        lp = StoreTemp('OUTPUT',
               Apply([(None, AttIndex(1))],
                 ProjectingJoin(expression.EQ(AttIndex(0), AttIndex(3)),
                   Scan(self.x_key, self.x_scheme),
                   Scan(self.x_key, self.x_scheme),
                   [AttIndex(i) for i in xrange(2 * len(self.x_scheme))])))  # noqa

        self.assertIsInstance(lp.input.input, ProjectingJoin)
        self.assertEquals(2 * len(self.x_scheme),
                          len(lp.input.input.scheme()))

        pp = self.logical_to_physical(lp)
        self.assertIsInstance(pp.input, MyriaSplitConsumer)
        proj_join = pp.input.input.input
        self.assertIsInstance(proj_join, ProjectingJoin)
        self.assertEquals(1, len(proj_join.scheme()))
        self.assertEquals(2, len(proj_join.left.scheme()))
        self.assertEquals(1, len(proj_join.right.scheme()))

        expected = collections.Counter(
            [(b,)
             for (a, b, c) in self.x_data
             for (d, e, f) in self.x_data
             if a == d])
        self.db.evaluate(pp)
        result = self.db.get_temp_table('OUTPUT')
        self.assertEquals(result, expected)

    def test_push_selects_apply(self):
        """Test pushing selections through apply."""
        lp = StoreTemp('OUTPUT',
               Select(expression.LTEQ(AttRef("c"), AttRef("a")),
                 Select(expression.LTEQ(AttRef("b"), AttRef("c")),
                   Apply([('b', AttIndex(1)),
                          ('c', AttIndex(2)),
                          ('a', AttIndex(0))],
                         Scan(self.x_key, self.x_scheme)))))  # noqa

        expected = collections.Counter(
            [(b, c, a) for (a, b, c) in self.x_data if c <= a and b <= c])

        self.assertEquals(self.get_count(lp, Select), 2)
        self.assertEquals(self.get_count(lp, Scan), 1)
        self.assertIsInstance(lp.input, Select)

        pp = self.logical_to_physical(lp)
        self.assertIsInstance(pp.input, Apply)
        self.assertEquals(self.get_count(pp, Select), 1)

        self.db.evaluate(pp)
        result = self.db.get_temp_table('OUTPUT')
        self.assertEquals(result, expected)

    def test_push_selects_groupby(self):
        """Test pushing selections through groupby."""
        lp = StoreTemp('OUTPUT',
               Select(expression.LTEQ(AttRef("c"), AttRef("a")),
                 Select(expression.LTEQ(AttRef("b"), AttRef("c")),
                   GroupBy([AttIndex(1), AttIndex(2), AttIndex(0)],
                           [expression.COUNTALL()],
                           Scan(self.x_key, self.x_scheme)))))  # noqa

        expected = collections.Counter(
            [(b, c, a) for (a, b, c) in self.x_data if c <= a and b <= c])
        expected = collections.Counter(k + (v,) for k, v in expected.items())

        self.assertEquals(self.get_count(lp, Select), 2)
        self.assertEquals(self.get_count(lp, Scan), 1)
        self.assertIsInstance(lp.input, Select)

        pp = self.logical_to_physical(lp)
        self.assertIsInstance(pp.input, MyriaSplitConsumer)
        self.assertIsInstance(pp.input.input.input, GroupBy)
        self.assertEquals(self.get_count(pp, Select), 1)

        self.db.evaluate(pp)
        result = self.db.get_temp_table('OUTPUT')
        self.assertEquals(result, expected)

    def test_noop_apply_removed(self):
        lp = StoreTemp('OUTPUT',
               Apply([(None, AttIndex(1))],
                 ProjectingJoin(expression.EQ(AttIndex(0), AttIndex(3)),
                   Scan(self.x_key, self.x_scheme),
                   Scan(self.x_key, self.x_scheme),
                   [AttIndex(i) for i in xrange(2 * len(self.x_scheme))])))  # noqa

        self.assertIsInstance(lp.input, Apply)
        lp_scheme = lp.scheme()

        pp = self.logical_to_physical(lp)
        self.assertNotIsInstance(pp.input, Apply)
        self.assertEquals(lp_scheme, pp.scheme())

    def test_not_noop_apply_not_removed(self):
        lp = StoreTemp('OUTPUT',
               Apply([('hi', AttIndex(1))],
                 ProjectingJoin(expression.EQ(AttIndex(0), AttIndex(3)),
                   Scan(self.x_key, self.x_scheme),
                   Scan(self.x_key, self.x_scheme),
                   [AttIndex(i) for i in xrange(2 * len(self.x_scheme))])))  # noqa

        self.assertIsInstance(lp.input, Apply)
        lp_scheme = lp.scheme()

        pp = self.logical_to_physical(lp)
        self.assertIsInstance(pp.input, Apply)
        self.assertEquals(lp_scheme, pp.scheme())

    def test_extract_join(self):
        """Extract a join condition from the middle of complex select."""
        s = expression.AND(expression.LTEQ(AttRef("e"), AttRef("f")),
                           expression.AND(
                               expression.EQ(AttRef("c"), AttRef("d")),
                               expression.GT(AttRef("a"), AttRef("b"))))

        lp = StoreTemp('OUTPUT', Select(s, CrossProduct(
            Scan(self.x_key, self.x_scheme),
            Scan(self.y_key, self.y_scheme))))

        self.assertEquals(self.get_num_select_conjuncs(lp), 3)

        pp = self.logical_to_physical(lp)

        # non-equijoin conditions should get pushed separately below the join
        self.assertIsInstance(pp.input, MyriaSplitConsumer)
        self.assertIsInstance(pp.input.input.input, Join)
        self.assertEquals(self.get_count(pp, CrossProduct), 0)
        self.assertEquals(self.get_count(pp, Select), 2)

        self.db.evaluate(pp)
        result = self.db.get_temp_table('OUTPUT')
        expected = collections.Counter(
            [(a, b, c, d, e, f) for (a, b, c) in self.x_data
             for (d, e, f) in self.y_data if a > b and e <= f and c == d])
        self.assertEquals(result, expected)

    def test_multi_condition_join(self):
        s = expression.AND(expression.EQ(AttRef("c"), AttRef("d")),
                           expression.EQ(AttRef("a"), AttRef("f")))

        lp = StoreTemp('OUTPUT', Select(s, CrossProduct(
            Scan(self.x_key, self.x_scheme),
            Scan(self.y_key, self.y_scheme))))

        self.assertEquals(self.get_num_select_conjuncs(lp), 2)

        pp = self.logical_to_physical(lp)
        self.assertEquals(self.get_count(pp, CrossProduct), 0)
        self.assertEquals(self.get_count(pp, Select), 0)

        expected = collections.Counter(
            [(a, b, c, d, e, f) for (a, b, c) in self.x_data
             for (d, e, f) in self.y_data if a == f and c == d])
        self.db.evaluate(pp)
        result = self.db.get_temp_table('OUTPUT')
        self.assertEquals(result, expected)

    def test_multiway_join_left_deep(self):

        query = """
        T = SCAN(public:adhoc:Z);
        U = [FROM T AS T1, T AS T2, T AS T3
             WHERE T1.dst==T2.src AND T2.dst==T3.src
             EMIT T1.src AS x, T3.dst AS y];
        STORE(U, OUTPUT);
        """

        lp = self.get_logical_plan(query)
        self.assertEquals(self.get_count(lp, CrossProduct), 2)
        self.assertEquals(self.get_count(lp, Join), 0)

        pp = self.logical_to_physical(lp)
        self.assertEquals(self.get_count(pp, CrossProduct), 0)
        self.assertEquals(self.get_count(pp, Join), 2)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 4)
        self.assertEquals(self.get_count(pp, NaryJoin), 0)
        self.assertEquals(self.get_count(pp, MyriaHyperCubeShuffleProducer), 0)

        self.db.evaluate(pp)
        result = self.db.get_table('OUTPUT')
        expected = collections.Counter(
            [(s1, d3) for (s1, d1) in self.z_data.elements()
             for (s2, d2) in self.z_data.elements()
             for (s3, d3) in self.z_data.elements() if d1 == s2 and d2 == s3])
        self.assertEquals(result, expected)

    def test_multiway_join_hyper_cube(self):

        query = """
        T = SCAN(public:adhoc:Z);
        U = [FROM T AS T1, T AS T2, T AS T3
             WHERE T1.dst==T2.src AND T2.dst==T3.src
             EMIT T1.src AS x, T3.dst AS y];
        STORE(U, OUTPUT);
        """

        lp = self.get_logical_plan(query)
        self.assertEquals(self.get_count(lp, CrossProduct), 2)
        self.assertEquals(self.get_count(lp, Join), 0)

        pp = self.logical_to_physical(lp, hypercube=True)
        self.assertEquals(self.get_count(pp, CrossProduct), 0)
        self.assertEquals(self.get_count(pp, Join), 0)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 0)
        self.assertEquals(self.get_count(pp, NaryJoin), 1)
        self.assertEquals(self.get_count(pp, MyriaHyperCubeShuffleProducer), 3)

        self.db.evaluate(pp)
        result = self.db.get_table('OUTPUT')
        expected = collections.Counter(
            [(s1, d3) for (s1, d1) in self.z_data.elements()
             for (s2, d2) in self.z_data.elements()
             for (s3, d3) in self.z_data.elements() if d1 == s2 and d2 == s3])
        self.assertEquals(result, expected)

    def test_hyper_cube_tie_breaking_heuristic(self):
        query = """
        T = SCAN(public:adhoc:Z);
        U = [FROM T AS T1, T AS T2, T AS T3, T AS T4
             WHERE T1.dst=T2.src AND T2.dst=T3.src AND
                   T3.dst=T4.src AND T4.dst=T1.src
             EMIT T1.src AS x, T3.dst AS y];
        STORE(U, OUTPUT);
        """
        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp, hypercube=True)

        def get_max_dim_size(_op):
            if isinstance(_op, MyriaHyperCubeShuffleProducer):
                yield max(_op.hyper_cube_dimensions)

        # the max hypercube dim size will be 8, e.g (1, 8, 1, 8) without
        # tie breaking heuristic, now it is (2, 4, 2, 4)
        self.assertTrue(max(pp.postorder(get_max_dim_size)) <= 4)

    def test_naryjoin_merge(self):
        query = """
        T1 = scan(public:adhoc:Z);
        T2 = [from T1 emit count(dst) as dst, src];
        T3 = scan(public:adhoc:Z);
        twohop = [from T1, T2, T3
                  where T1.dst = T2.src and T2.dst = T3.src
                  emit *];
        store(twohop, anothertwohop);
        """
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        lp = self.processor.get_logical_plan()
        pp = self.logical_to_physical(lp, hypercube=True)
        self.assertEquals(self.get_count(pp, NaryJoin), 0)

    def test_right_deep_join(self):
        """Test pushing a selection into a right-deep join tree.

        Myrial doesn't emit these, so we need to cook up a plan by hand."""

        s = expression.AND(expression.EQ(AttIndex(1), AttIndex(2)),
                           expression.EQ(AttIndex(3), AttIndex(4)))

        lp = Apply([('x', AttIndex(0)), ('y', AttIndex(5))],
                   Select(s,
                          CrossProduct(Scan(self.z_key, self.z_scheme),
                                       CrossProduct(
                                           Scan(self.z_key, self.z_scheme),
                                           Scan(self.z_key, self.z_scheme)))))
        lp = StoreTemp('OUTPUT', lp)

        self.assertEquals(self.get_count(lp, CrossProduct), 2)

        pp = self.logical_to_physical(lp)
        self.assertEquals(self.get_count(pp, CrossProduct), 0)

        self.db.evaluate(pp)

        result = self.db.get_temp_table('OUTPUT')
        expected = collections.Counter(
            [(s1, d3) for (s1, d1) in self.z_data.elements()
             for (s2, d2) in self.z_data.elements()
             for (s3, d3) in self.z_data.elements() if d1 == s2 and d2 == s3])
        self.assertEquals(result, expected)

    def test_explicit_shuffle(self):
        """Test of a user-directed partition operation."""

        query = """
        T = SCAN(public:adhoc:X);
        STORE(T, OUTPUT, [$2, b]);
        """
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        lp = self.processor.get_logical_plan()

        self.assertEquals(self.get_count(lp, Shuffle), 1)

        for op in lp.walk():
            if isinstance(op, Shuffle):
                self.assertEquals(op.columnlist, [AttIndex(2), AttIndex(1)])

    def test_shuffle_before_distinct(self):
        query = """
        T = DISTINCT(SCAN(public:adhoc:Z));
        STORE(T, OUTPUT);
        """

        pp = self.get_physical_plan(query)
        self.assertEquals(self.get_count(pp, Distinct), 2)  # distributed
        first = True
        for op in pp.walk():
            if isinstance(op, Distinct):
                self.assertIsInstance(op.input, MyriaShuffleConsumer)
                self.assertIsInstance(op.input.input, MyriaShuffleProducer)
                break

    def test_shuffle_before_difference(self):
        query = """
        T = DIFF(SCAN(public:adhoc:Z), SCAN(public:adhoc:Z));
        STORE(T, OUTPUT);
        """

        pp = self.get_physical_plan(query)
        self.assertEquals(self.get_count(pp, Difference), 1)
        for op in pp.walk():
            if isinstance(op, Difference):
                self.assertIsInstance(op.left, MyriaShuffleConsumer)
                self.assertIsInstance(op.left.input, MyriaShuffleProducer)
                self.assertIsInstance(op.right, MyriaShuffleConsumer)
                self.assertIsInstance(op.right.input, MyriaShuffleProducer)

    def test_bug_240_broken_remove_unused_columns_rule(self):
        query = """
        particles = empty(nowGroup:int, timestep:int, grp:int);

        haloTable1 = [from particles as P
                      emit P.nowGroup,
                           (P.timestep+P.grp) as halo,
                           count(*) as totalParticleCount];

        haloTable2 = [from haloTable1 as H, particles as P
                      where H.nowGroup = P.nowGroup
                      emit *];
        store(haloTable2, OutputTemp);
        """

        # This is it -- just test that we can get the physical plan and
        # compile to JSON. See https://github.com/uwescience/raco/issues/240
        pp = self.execute_query(query, output='OutputTemp')

    def test_broadcast_cardinality_right(self):
        # x and y have the same cardinality, z is smaller
        query = """
        x = scan({x});
        y = scan({y});
        z = scan({z});
        out = [from x, z emit *];
        store(out, OUTPUT);
        """.format(x=self.x_key, y=self.y_key, z=self.z_key)

        pp = self.get_physical_plan(query)
        counter = 0
        for op in pp.walk():
            if isinstance(op, CrossProduct):
                counter += 1
                self.assertIsInstance(op.right, MyriaBroadcastConsumer)
        self.assertEquals(counter, 1)

    def test_broadcast_cardinality_left(self):
        # x and y have the same cardinality, z is smaller
        query = """
        x = scan({x});
        y = scan({y});
        z = scan({z});
        out = [from z, y emit *];
        store(out, OUTPUT);
        """.format(x=self.x_key, y=self.y_key, z=self.z_key)

        pp = self.get_physical_plan(query)
        counter = 0
        for op in pp.walk():
            if isinstance(op, CrossProduct):
                counter += 1
                self.assertIsInstance(op.left, MyriaBroadcastConsumer)
        self.assertEquals(counter, 1)

    def test_broadcast_cardinality_with_agg(self):
        # x and y have the same cardinality, z is smaller
        query = """
        x = scan({x});
        y = countall(scan({y}));
        z = scan({z});
        out = [from y, z emit *];
        store(out, OUTPUT);
        """.format(x=self.x_key, y=self.y_key, z=self.z_key)

        pp = self.get_physical_plan(query)
        counter = 0
        for op in pp.walk():
            if isinstance(op, CrossProduct):
                counter += 1
                self.assertIsInstance(op.left, MyriaBroadcastConsumer)
        self.assertEquals(counter, 1)

    def test_relation_cardinality(self):
        query = """
        x = scan({x});
        out = [from x as x1, x as x2 emit *];
        store(out, OUTPUT);
        """.format(x=self.x_key)
        lp = self.get_logical_plan(query)
        self.assertIsInstance(lp, Sequence)
        self.assertEquals(1, len(lp.children()))
        self.assertEquals(sum(self.x_data.values()) ** 2,
                          lp.children()[0].num_tuples())

    def test_relation_physical_cardinality(self):
        query = """
        x = scan({x});
        out = [from x as x1, x as x2 emit *];
        store(out, OUTPUT);
        """.format(x=self.x_key)

        pp = self.get_physical_plan(query)
        self.assertEquals(sum(self.x_data.values()) ** 2,
                          pp.num_tuples())

    def test_catalog_cardinality(self):
        self.assertEquals(sum(self.x_data.values()),
                          self.db.num_tuples(self.x_key))
        self.assertEquals(sum(self.y_data.values()),
                          self.db.num_tuples(self.y_key))
        self.assertEquals(sum(self.z_data.values()),
                          self.db.num_tuples(self.z_key))

    def test_groupby_to_distinct(self):
        query = """
        x = scan({x});
        y = select $0, count(*) from x;
        z = select $0 from y;
        store(z, OUTPUT);
        """.format(x=self.x_key)

        lp = self.get_logical_plan(query)
        self.assertEquals(self.get_count(lp, GroupBy), 1)
        self.assertEquals(self.get_count(lp, Distinct), 0)

        pp = self.logical_to_physical(copy.deepcopy(lp))
        self.assertEquals(self.get_count(pp, GroupBy), 0)
        self.assertEquals(self.get_count(pp, Distinct), 2)  # distributed

        self.assertEquals(self.db.evaluate(lp), self.db.evaluate(pp))

    def test_groupby_to_lesser_groupby(self):
        query = """
        x = scan({x});
        y = select $0, count(*), sum($1) from x;
        z = select $0, $2 from y;
        store(z, OUTPUT);
        """.format(x=self.x_key)

        lp = self.get_logical_plan(query)
        self.assertEquals(self.get_count(lp, GroupBy), 1)
        for op in lp.walk():
            if isinstance(op, GroupBy):
                self.assertEquals(len(op.grouping_list), 1)
                self.assertEquals(len(op.aggregate_list), 2)

        pp = self.logical_to_physical(copy.deepcopy(lp))
        self.assertEquals(self.get_count(pp, GroupBy), 2)  # distributed
        for op in pp.walk():
            if isinstance(op, GroupBy):
                self.assertEquals(len(op.grouping_list), 1)
                self.assertEquals(len(op.aggregate_list), 1)

        self.assertEquals(self.db.evaluate(lp), self.db.evaluate(pp))

    def __run_uda_test(self, uda_state=None):
        scan = Scan(self.x_key, self.x_scheme)

        init_ex = expression.NumericLiteral(0)
        update_ex = expression.PLUS(expression.NamedStateAttributeRef("value"),
                                    AttIndex(1))
        emit_ex = expression.UdaAggregateExpression(
            expression.NamedStateAttributeRef("value"), uda_state)
        statemods = [StateVar("value", init_ex, update_ex)]

        log_gb = GroupBy([AttIndex(0)], [emit_ex], scan, statemods)

        lp = StoreTemp('OUTPUT', log_gb)
        pp = self.logical_to_physical(copy.deepcopy(lp))

        self.db.evaluate(lp)
        log_result = self.db.get_temp_table('OUTPUT')

        self.db.delete_temp_table('OUTPUT')
        self.db.evaluate(pp)
        phys_result = self.db.get_temp_table('OUTPUT')

        self.assertEquals(log_result, phys_result)
        self.assertEquals(len(log_result), 15)

        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 1)
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 1)

        return pp

    def test_non_decomposable_uda(self):
        """Test that optimization preserves the value of a non-decomposable UDA
        """
        pp = self.__run_uda_test()

        for op in pp.walk():
            if isinstance(op, MyriaShuffleProducer):
                self.assertEquals(op.hash_columns, [AttIndex(0)])
                self.assertEquals(self.get_count(op, GroupBy), 0)

    def test_decomposable_uda(self):
        """Test that optimization preserves the value of decomposable UDAs"""
        lemits = [expression.UdaAggregateExpression(
                  expression.NamedStateAttributeRef("value"))]
        remits = copy.deepcopy(lemits)

        init_ex = expression.NumericLiteral(0)
        update_ex = expression.PLUS(expression.NamedStateAttributeRef("value"),
                                    AttIndex(1))
        lstatemods = [StateVar("value", init_ex, update_ex)]
        rstatemods = copy.deepcopy(lstatemods)

        uda_state = expression.DecomposableAggregateState(
            lemits, lstatemods, remits, rstatemods)
        pp = self.__run_uda_test(uda_state)

        self.assertEquals(self.get_count(pp, GroupBy), 2)

        for op in pp.walk():
            if isinstance(op, MyriaShuffleProducer):
                self.assertEquals(op.hash_columns, [AttIndex(0)])
                self.assertEquals(self.get_count(op, GroupBy), 1)

    def test_successful_append(self):
        """Insert an append if storing a relation into itself with a
        UnionAll."""
        query = """
        x = scan({x});
        y = select $0 from x;
        y2 = select $1 from x;
        y = y+y2;
        store(y, OUTPUT);
        """.format(x=self.x_key)

        lp = self.get_logical_plan(query, apply_chaining=False)
        self.assertEquals(self.get_count(lp, ScanTemp), 5)
        self.assertEquals(self.get_count(lp, StoreTemp), 4)
        self.assertEquals(self.get_count(lp, AppendTemp), 0)
        self.assertEquals(self.get_count(lp, Store), 1)
        self.assertEquals(self.get_count(lp, Scan), 1)

        pp = self.logical_to_physical(copy.deepcopy(lp))
        self.assertEquals(self.get_count(pp, ScanTemp), 4)
        self.assertEquals(self.get_count(pp, StoreTemp), 3)
        self.assertEquals(self.get_count(pp, AppendTemp), 1)
        self.assertEquals(self.get_count(pp, Store), 1)
        self.assertEquals(self.get_count(pp, Scan), 1)

        self.assertEquals(self.db.evaluate(lp), self.db.evaluate(pp))

    def test_failed_append(self):
        """Do not insert an append when the tuples to be appended
        depend on the relation itself."""

        # NB test in both the left and right directions
        # left: y = y + y2
        # right: y = y2 + y
        query = """
        x = scan({x});
        y = select $0, $1 from x;
        t = empty(a:int);
        y2 = select $1, $1 from y;
        y = y+y2;
        t = empty(a:int);
        y3 = select $1, $1 from y;
        y = y3+y;
        s = empty(a:int);
        store(y, OUTPUT);
        """.format(x=self.x_key)

        lp = self.get_logical_plan(query, dead_code_elimination=False)
        self.assertEquals(self.get_count(lp, AppendTemp), 0)

        # No AppendTemp
        pp = self.logical_to_physical(copy.deepcopy(lp))
        self.assertEquals(self.get_count(pp, AppendTemp), 0)

        self.assertEquals(self.db.evaluate(lp), self.db.evaluate(pp))

    def test_push_work_into_sql(self):
        """Test generation of MyriaQueryScan operator for query with
        projects"""
        query = """
        r3 = scan({x});
        intermediate = select a, c from r3;
        store(intermediate, OUTPUT);
        """.format(x=self.x_key)

        pp = self.get_physical_plan(query, push_sql=True)
        self.assertEquals(self.get_count(pp, Operator), 2)
        self.assertTrue(isinstance(pp.input, MyriaQueryScan))

        expected = collections.Counter([(a, c) for (a, b, c) in self.x_data])

        self.db.evaluate(pp)
        result = self.db.get_table('OUTPUT')
        self.assertEquals(result, expected)

    def test_push_work_into_sql_2(self):
        """Test generation of MyriaQueryScan operator for query with projects
        and a filter"""
        query = """
        r3 = scan({x});
        intermediate = select a, c from r3 where b < 5;
        store(intermediate, OUTPUT);
        """.format(x=self.x_key)

        pp = self.get_physical_plan(query, push_sql=True)
        self.assertEquals(self.get_count(pp, Operator), 2)
        self.assertTrue(isinstance(pp.input, MyriaQueryScan))

        expected = collections.Counter([(a, c)
                                        for (a, b, c) in self.x_data
                                        if b < 5])

        self.db.evaluate(pp)
        result = self.db.get_table('OUTPUT')
        self.assertEquals(result, expected)

    def test_no_push_when_shuffle(self):
        """When data is not co-partitioned, the join should not be pushed."""
        query = """
        r3 = scan({x});
        s3 = scan({y});
        intermediate = select r3.a, s3.f from r3, s3 where r3.b=s3.e;
        store(intermediate, OUTPUT);
        """.format(x=self.x_key, y=self.y_key)

        pp = self.get_physical_plan(query, push_sql=True)
        # Join is not pushed
        self.assertEquals(self.get_count(pp, Join), 1)
        # The projections are pushed into the QueryScan
        self.assertEquals(self.get_count(pp, MyriaQueryScan), 2)
        # We should not need any Apply since there is no rename and no other
        # project.
        self.assertEquals(self.get_count(pp, Apply), 0)

        expected = collections.Counter([(a, f)
                                        for (a, b, c) in self.x_data
                                        for (d, e, f) in self.y_data
                                        if b == e])

        self.db.evaluate(pp)
        result = self.db.get_table('OUTPUT')
        self.assertEquals(result, expected)

    def test_no_push_when_random(self):
        """Selection with RANDOM() doesn't push through joins"""
        query = """
        r = scan({x});
        s = scan({y});
        t = [from r,s where random()*10 > .3 emit *];
        store(t, OUTPUT);
        """.format(x=self.x_key, y=self.y_key)

        lp = self.get_logical_plan(query)
        self.assertEquals(self.get_count(lp, Select), 1)
        self.assertEquals(self.get_count(lp, CrossProduct), 1)

        pp = self.logical_to_physical(lp)
        self.assertEquals(self.get_count(pp, Select), 1)
        self.assertEquals(self.get_count(pp, CrossProduct), 1)
        # The selection should happen after the cross product
        for op in pp.walk():
            if isinstance(op, Select):
                self.assertIsInstance(op.input, MyriaSplitConsumer)
                self.assertIsInstance(op.input.input.input, CrossProduct)

    def test_partitioning_from_shuffle(self):
        """Store will know the partitioning of a shuffled relation"""
        query = """
        r = scan({x});
        store(r, OUTPUT);
        """.format(x=self.x_key)

        lp = self.get_logical_plan(query)

        # insert a shuffle
        tail = lp.args[0].input
        lp.args[0].input = Shuffle(tail, [AttIndex(0)])

        pp = self.logical_to_physical(lp)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 1)
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 1)
        self.assertEquals(pp.partitioning().hash_partitioned,
                          frozenset([AttIndex(0)]))

    def test_partitioning_from_scan(self):
        """Store will know the partitioning of a partitioned store relation"""
        query = """
        r = scan({part});
        store(r, OUTPUT);
        """.format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp)

        self.assertEquals(pp.partitioning().hash_partitioned,
                          self.part_partition.hash_partitioned)

    def test_repartitioning(self):
        """Shuffle repartition a partitioned relation"""
        query = """
        r = scan({part});
        store(r, OUTPUT);
        """.format(part=self.part_key)

        lp = self.get_logical_plan(query)

        # insert a shuffle
        tail = lp.args[0].input
        lp.args[0].input = Shuffle(tail, [AttIndex(2)])

        pp = self.logical_to_physical(lp)

        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 1)
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 1)
        self.assertEquals(pp.partitioning().hash_partitioned,
                          frozenset([AttIndex(2)]))

    def test_remove_shuffle(self):
        """No shuffle for hash join needed when the input is partitioned"""
        query = """
        r = scan({part});
        s = scan({part});
        t = select * from r, s where r.h = s.h;
        store(t, OUTPUT);""".format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp)
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 0)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 0)

    def test_do_not_remove_shuffle_left(self):
        """Shuffle for hash join needed when the input is partitioned wrong"""
        query = """
        r = scan({part});
        s = scan({part});
        t = select * from r, s where r.i = s.h;
        store(t, OUTPUT);""".format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp)
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 1)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 1)

    def test_do_not_remove_shuffle_both(self):
        """Shuffle for hash join needed when the input is partitioned wrong"""
        query = """
        r = scan({part});
        s = scan({part});
        t = select * from r, s where r.i = s.i;
        store(t, OUTPUT);""".format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp)
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 2)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 2)

    def test_apply_removes_partitioning(self):
        """Projecting out any partitioned attribute
        eliminates partition info"""

        query = """
        r = scan({part});
        s = select g,i from r;
        store(s, OUTPUT);
        """.format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp)

        self.assertEquals(pp.partitioning().hash_partitioned,
                          frozenset())

    def test_apply_maintains_partitioning(self):
        """Projecting out non-partitioned attributes
        does not eliminate partition info"""

        query = """
        r = scan({part});
        s = select h, i from r;
        store(s, OUTPUT);
        """.format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp)

        self.assertEquals(pp.partitioning().hash_partitioned,
                          frozenset([AttIndex(0)]))

    def test_swapping_apply_maintains_partitioning(self):
        """Projecting out non-partitioned attributes
        does not eliminate partition info, even for swaps"""

        query = """
        r = scan({part});
        s = select i, h from r;
        store(s, OUTPUT);
        """.format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp)

        self.assertEquals(pp.partitioning().hash_partitioned,
                          frozenset([AttIndex(1)]))

    def test_projecting_join_maintains_partitioning(self):
        """Projecting join: projecting out non-partitioned attributes
        does not eliminate partition info.
        """

        query = """
        r = scan({part});
        s = scan({part});
        t = select r.h, r.i, s.h, s.i from r, s where r.h = s.h;
        store(t, OUTPUT);""".format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp)

        # shuffles should be removed
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 0)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 0)

        # TODO: this test case forces conservative behavior
        # (in general, info could be h($0) && h($2)
        self.assertEquals(pp.partitioning().hash_partitioned,
                          frozenset([AttIndex(0)]))

    def test_no_shuffle_for_partitioned_distinct(self):
        """Do not shuffle for Distinct if already partitioned"""

        query = """
        r = scan({part});
        t = select distinct r.h from r;
        store(t, OUTPUT);""".format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp)

        # shuffles should be removed and distinct not decomposed into two
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 0)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 0)
        self.assertEquals(self.get_count(pp, MyriaDupElim), 1)

        self.db.evaluate(pp)
        result = self.db.get_table('OUTPUT')
        expected = dict([((h,), 1) for _, h, _ in self.part_data])
        self.assertEquals(result, expected)

    def test_no_shuffle_for_partitioned_groupby(self):
        """Do not shuffle for groupby if already partitioned"""

        query = """
        r = scan({part});
        t = select r.h, MIN(r.i) from r;
        store(t, OUTPUT);""".format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp)

        # shuffles should be removed and the groupby not decomposed into two
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 0)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 0)
        self.assertEquals(self.get_count(pp, MyriaGroupBy), 1)

    def test_partition_aware_groupby_into_sql(self):
        """No shuffle for groupby also causes it to be pushed into sql"""

        query = """
        r = scan({part});
        t = select r.h, MIN(r.i) from r;
        store(t, OUTPUT);""".format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp, push_sql=True,
                                      push_sql_grouping=True)

        # shuffles should be removed and the groupby not decomposed into two
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 0)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 0)

        # should be pushed
        self.assertEquals(self.get_count(pp, MyriaGroupBy), 0)
        self.assertEquals(self.get_count(pp, MyriaQueryScan), 1)

        self.db.evaluate(pp)
        result = self.db.get_table('OUTPUT')
        temp = dict([(h, sys.maxsize) for _, h, _ in self.part_data])
        for _, h, i in self.part_data:
            temp[h] = min(temp[h], i)
        expected = dict(((h, i), 1) for h, i in temp.items())

        self.assertEquals(result, expected)

    def test_partition_aware_distinct_into_sql(self):
        """No shuffle for distinct also causes it to be pushed into sql"""

        query = """
        r = scan({part});
        t = select distinct r.h from r;
        store(t, OUTPUT);""".format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp, push_sql=True)

        # shuffles should be removed and the groupby not decomposed into two
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 0)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 0)

        # should be pushed
        self.assertEquals(self.get_count(pp, MyriaGroupBy), 0)  # sanity
        self.assertEquals(self.get_count(pp, MyriaDupElim), 0)
        self.assertEquals(self.get_count(pp, MyriaQueryScan), 1)

        self.db.evaluate(pp)
        result = self.db.get_table('OUTPUT')
        expected = dict([((h,), 1) for _, h, _ in self.part_data])
        self.assertEquals(result, expected)

    def test_push_half_groupby_into_sql(self):
        """Push the first group by of decomposed group by into sql"""

        query = """
        r = scan({part});
        t = select r.i, MIN(r.h) from r;
        store(t, OUTPUT);""".format(part=self.part_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp, push_sql=True,
                                      push_sql_grouping=True)

        # wrong partition, so still has shuffle
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 1)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 1)

        # one group by should be pushed
        self.assertEquals(self.get_count(pp, MyriaGroupBy), 1)
        self.assertEquals(self.get_count(pp, MyriaQueryScan), 1)

        self.db.evaluate(pp)
        result = self.db.get_table('OUTPUT')
        temp = dict([(i, sys.maxsize) for _, _, i in self.part_data])
        for _, h, i in self.part_data:
            temp[i] = min(temp[i], h)
        expected = dict(((k, v), 1) for k, v in temp.items())

        self.assertEquals(result, expected)

    def _check_aggregate_functions_pushed(
            self,
            func,
            expected,
            override=False):
        if override:
            agg = func
        else:
            agg = "{func}(r.i)".format(func=func)

        query = """
        r = scan({part});
        t = select r.h, {agg} from r;
        store(t, OUTPUT);""".format(part=self.part_key, agg=agg)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp, push_sql=True,
                                      push_sql_grouping=True)

        self.assertEquals(self.get_count(pp, MyriaQueryScan), 1)

        for op in pp.walk():
            if isinstance(op, MyriaQueryScan):
                self.assertTrue(re.search(expected, op.sql))

    def test_aggregate_AVG_pushed(self):
        """AVG is translated properly for postgresql. This is
        a function not in SQLAlchemy"""
        self._check_aggregate_functions_pushed(
            aggregate.AVG.__name__, 'avg')

    def test_aggregate_STDDEV_pushed(self):
        """STDEV is translated properly for postgresql. This is
        a function that is named differently in Raco and postgresql"""
        self._check_aggregate_functions_pushed(
            aggregate.STDEV.__name__, 'stddev_samp')

    def test_aggregate_COUNTALL_pushed(self):
        """COUNTALL is translated properly for postgresql. This is
        a function that is expressed differently in Raco and postgresql"""

        # MyriaL parses count(*) to Raco COUNTALL. And COUNTALL
        # should currently (under the no nulls semantics of Raco/Myria)
        # translate to COUNT(something)
        self._check_aggregate_functions_pushed(
            'count(*)', r'count[(][a-zA-Z.]+[)]', True)

    def test_debroadcast_broadcasted_relation(self):
        """Test that a shuffle over a broadcasted relation debroadcasts it"""
        query = """
        a = scan({broad});
        b = select j,k,l from a where j < 5;
        store(b, OUTPUT, [j, k]);""".format(broad=self.broad_key)

        pp = self.get_physical_plan(query)

        def find_scan(_op):
            if isinstance(_op, MyriaQueryScan) or isinstance(_op, MyriaScan):
                if _op._debroadcast:
                    yield True
                else:
                    yield False
            else:
                yield False

        self.assertEquals(self.get_count(pp, MyriaSelect), 1)
        self.assertTrue(any(pp.postorder(find_scan)))

    def test_broadcast_store(self):
        query = """
        r = scan({X});
        store(r, OUTPUT, broadcast());
        """.format(X=self.x_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp)

        self.assertEquals(self.get_count(pp, MyriaBroadcastConsumer), 1)
        self.assertEquals(self.get_count(pp, MyriaBroadcastProducer), 1)
        self.assertEquals(pp.partitioning().broadcasted,
                          RepresentationProperties(
            broadcasted=True).broadcasted)

    def test_broadcast_join(self):
        query = """
        b = scan({broad});
        x = scan({X});
        o = select * from b, x where b.j==x.a;
        store(o, OUTPUT);
        """.format(X=self.x_key, broad=self.broad_key)

        lp = self.get_logical_plan(query)
        pp = self.logical_to_physical(lp)

        self.assertEquals(self.get_count(pp, MyriaBroadcastProducer), 0)
        self.assertEquals(self.get_count(pp, MyriaBroadcastConsumer), 0)
        self.assertEquals(self.get_count(pp, MyriaShuffleProducer), 1)
        self.assertEquals(self.get_count(pp, MyriaShuffleConsumer), 1)
        self.assertEquals(pp.partitioning().broadcasted,
                          RepresentationProperties().broadcasted)

    def test_flatten_unionall(self):
        """Test flattening a chain of UnionAlls"""
        query = """
        X = scan({x});
        a = (select $0 from X) + [from X emit $0] + [from X emit $1];
        store(a, a);
        """.format(x=self.x_key)
        lp = self.get_logical_plan(query)
        # should be UNIONAll([UNIONAll([expr_1, expr_2]), expr_3])
        self.assertEquals(self.get_count(lp, UnionAll), 2)
        pp = self.logical_to_physical(lp)
        # should be UNIONALL([expr_1, expr_2, expr_3])
        self.assertEquals(self.get_count(pp, MyriaUnionAll), 1)

    def list_ops_in_json(self, plan, type):
        ops = []
        for p in plan['plan']['plans']:
            for frag in p['fragments']:
                for op in frag['operators']:
                    if op['opType'] == type:
                        ops.append(op)
        return ops

    def test_cc(self):
        """Test Connected Components"""
        query = """
        E = scan(public:adhoc:Z);
        V = select distinct E.src as x from E;
        do
            CC = [nid, MIN(cid) as cid] <-
                 [from V emit V.x as nid, V.x as cid] +
                 [from E, CC where E.src = CC.nid emit E.dst as nid, CC.cid];
        until convergence pull_idb;
        store(CC, CC);
        """
        lp = self.get_logical_plan(query, async_ft='REJOIN')
        pp = self.logical_to_physical(lp, async_ft='REJOIN')
        for op in pp.children():
            for child in op.children():
                if isinstance(child, MyriaIDBController):
                    # for checking rule RemoveSingleSplit
                    assert not isinstance(op, MyriaSplitProducer)
        plan = compile_to_json(query, lp, pp, 'myrial', async_ft='REJOIN')

        joins = [op for op in pp.walk()
                 if isinstance(op, MyriaSymmetricHashJoin)]
        assert len(joins) == 1
        assert joins[0].pull_order_policy == 'RIGHT'
        self.assertEquals(plan['ftMode'], 'REJOIN')
        idbs = self.list_ops_in_json(plan, 'IDBController')
        self.assertEquals(len(idbs), 1)
        self.assertEquals(idbs[0]['argState']['type'], 'KeepMinValue')
        self.assertEquals(idbs[0]['sync'], False)  # default value: async
        sps = self.list_ops_in_json(plan, 'ShuffleProducer')
        assert any(sp['argBufferStateType']['type'] == 'KeepMinValue'
                   for sp in sps if 'argBufferStateType' in sp and
                   sp['argBufferStateType'] is not None)

    def test_lca(self):
        """Test LCA"""
        query = """
        Cite = scan(public:adhoc:X);
        Paper = scan(public:adhoc:Y);
        do
        Ancestor = [a,b,MIN(dis) as dis] <- [from Cite emit a, b, 1 as dis] +
                [from Ancestor, Cite
                 where Ancestor.b = Cite.a
                 emit Ancestor.a, Cite.b, Ancestor.dis+1];
        LCA = [pid1,pid2,LEXMIN(dis,yr,anc)] <-
                [from Ancestor as A1, Ancestor as A2, Paper
                 where A1.b = A2.b and A1.b = Paper.d and A1.a < A2.a
                 emit A1.a as pid1, A2.a as pid2,
                 greater(A1.dis, A2.dis) as dis,
                 Paper.e as yr, A1.b as anc];
        until convergence sync;
        store(LCA, LCA);
        """
        lp = self.get_logical_plan(query, async_ft='REJOIN')
        pp = self.logical_to_physical(lp, async_ft='REJOIN')
        plan = compile_to_json(query, lp, pp, 'myrial', async_ft='REJOIN')
        idbs = self.list_ops_in_json(plan, 'IDBController')
        self.assertEquals(len(idbs), 2)
        self.assertEquals(idbs[0]['argState']['type'], 'KeepMinValue')
        self.assertEquals(idbs[1]['argState']['type'], 'KeepMinValue')
        self.assertEquals(len(idbs[1]['argState']['valueColIndices']), 3)
        self.assertEquals(idbs[0]['sync'], True)
        self.assertEquals(idbs[1]['sync'], True)

    def test_galaxy_evolution(self):
        """Test Galaxy Evolution"""
        query = """
        GoI = scan(public:adhoc:X);
        Particles = scan(public:adhoc:Y);
        do
        Edges = [time,gid1,gid2,COUNT(*) as num] <-
                [from Particles as P1, Particles as P2, Galaxies
                where P1.d = P2.d and P1.f+1 = P2.f and
                      P1.f = Galaxies.time and Galaxies.gid = P1.e
                emit P1.f as time, P1.e as gid1, P2.e as gid2];
        Galaxies = [time, gid] <-
          [from GoI emit 1 as time, GoI.a as gid] +
          [from Galaxies, Edges
           where Galaxies.time = Edges.time and
           Galaxies.gid = Edges.gid1 and Edges.num >= 4
           emit Galaxies.time+1, Edges.gid2 as gid];
        until convergence async build_EDB;
        store(Galaxies, Galaxies);
        """
        lp = self.get_logical_plan(query, async_ft='REJOIN')
        for op in lp.walk():
            if isinstance(op, Select):
                # for checking rule RemoveEmptyFilter
                assert(op.condition is not None)
        pp = self.logical_to_physical(lp, async_ft='REJOIN')
        plan = compile_to_json(query, lp, pp, 'myrial', async_ft='REJOIN')
        joins = [op for op in pp.walk()
                 if isinstance(op, MyriaSymmetricHashJoin)]
        # The two joins for Edges
        assert len(
            [j for j in joins if j.pull_order_policy == 'LEFT_EOS']) == 2

        idbs = self.list_ops_in_json(plan, 'IDBController')
        self.assertEquals(len(idbs), 2)
        self.assertEquals(idbs[0]['argState']['type'], 'CountFilter')
        self.assertEquals(idbs[1]['argState']['type'], 'DupElim')
        self.assertEquals(idbs[0]['sync'], False)
        self.assertEquals(idbs[1]['sync'], False)

        super(OptimizerTest, self).new_processor()
        query = """
        GoI = scan(public:adhoc:X);
        Particles = scan(public:adhoc:Y);
        do
        Edges = [time,gid1,gid2,COUNT(*) as num] <-
                [from Particles as P1, Particles as P2, Galaxies
                where P1.d = P2.d and P1.f+1 = P2.f and
                      P1.f = Galaxies.time and Galaxies.gid = P1.e
                emit P1.f as time, P1.e as gid1, P2.e as gid2];
        Galaxies = [time, gid] <-
          [from GoI emit 1 as time, GoI.a as gid] +
          [from Galaxies, Edges
           where Galaxies.time = Edges.time and
           Galaxies.gid = Edges.gid1 and Edges.num > 3
           emit Galaxies.time+1, Edges.gid2 as gid];
        until convergence async build_EDB;
        store(Galaxies, Galaxies);
        """
        lp = self.get_logical_plan(query, async_ft='REJOIN')
        pp = self.logical_to_physical(lp, async_ft='REJOIN')
        plan_gt = compile_to_json(query, lp, pp, 'myrial', async_ft='REJOIN')
        idbs_gt = self.list_ops_in_json(plan_gt, 'IDBController')
        self.assertEquals(idbs_gt[0], idbs[0])

    def test_push_select_below_shuffle(self):
        """Test pushing selections below shuffles."""
        lp = StoreTemp('OUTPUT',
                       Select(expression.LTEQ(AttRef("a"), AttRef("b")),
                              Shuffle(
                                      Scan(self.x_key, self.x_scheme),
                                      [AttRef("a"), AttRef("b")], 'Hash')))  # noqa

        self.assertEquals(self.get_count(lp, StoreTemp), 1)
        self.assertEquals(self.get_count(lp, Select), 1)
        self.assertEquals(self.get_count(lp, Shuffle), 1)
        self.assertEquals(self.get_count(lp, Scan), 1)

        pp = self.logical_to_physical(lp)
        self.assertIsInstance(pp.input, MyriaShuffleConsumer)
        self.assertIsInstance(pp.input.input, MyriaShuffleProducer)
        self.assertIsInstance(pp.input.input.input, Select)
        self.assertIsInstance(pp.input.input.input.input, Scan)

    def test_insert_shuffle_after_filescan(self):
        """Test automatically inserting round-robin shuffle after FileScan."""
        query = """
        X = load('INPUT', csv(schema(a:int,b:int)));
        store(X, 'OUTPUT');"""

        lp = self.get_logical_plan(query)
        self.assertEquals(self.get_count(lp, Store), 1)
        self.assertEquals(self.get_count(lp, FileScan), 1)

        pp = self.logical_to_physical(lp)
        self.assertIsInstance(pp.input, MyriaShuffleConsumer)
        self.assertIsInstance(pp.input.input, MyriaShuffleProducer)
        self.assertEquals(pp.input.input.shuffle_type, 'RoundRobin')
        self.assertIsInstance(pp.input.input.input, FileScan)

    def test_elide_extra_shuffle_after_filescan(self):
        """Test eliding default round-robin shuffle after FileScan
        if shuffle is already present.
        """
        query = """
        X = load('INPUT', csv(schema(a:int,b:int)));
        store(X, 'OUTPUT', hash(a, b));"""

        lp = self.get_logical_plan(query)
        self.assertEquals(self.get_count(lp, Store), 1)
        self.assertEquals(self.get_count(lp, Shuffle), 1)
        self.assertEquals(self.get_count(lp, FileScan), 1)

        pp = self.logical_to_physical(lp)
        self.assertIsInstance(pp.input, MyriaShuffleConsumer)
        self.assertIsInstance(pp.input.input, MyriaShuffleProducer)
        self.assertEquals(pp.input.input.shuffle_type, 'Hash')
        self.assertIsInstance(pp.input.input.input, FileScan)

    def test_push_select_below_shuffle_inserted_for_filescan(self):
        """Test pushing selections below shuffles
        automatically inserted after FileScan.
        """
        query = """
        X = load('INPUT', csv(schema(a:int,b:int)));
        Y = select * from X where a > b;
        store(Y, 'OUTPUT');"""

        lp = self.get_logical_plan(query)
        self.assertEquals(self.get_count(lp, Store), 1)
        self.assertEquals(self.get_count(lp, Select), 1)
        self.assertEquals(self.get_count(lp, FileScan), 1)

        pp = self.logical_to_physical(lp)
        self.assertIsInstance(pp.input, MyriaShuffleConsumer)
        self.assertIsInstance(pp.input.input, MyriaShuffleProducer)
        self.assertIsInstance(pp.input.input.input, Select)
        self.assertIsInstance(pp.input.input.input.input, FileScan)
