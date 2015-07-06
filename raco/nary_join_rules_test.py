from nose.plugins.skip import SkipTest
import unittest
import algebra
from raco import RACompiler
from raco.catalog import FakeCatalog
from raco.backends import myria as myrialang


class testNaryJoin(unittest.TestCase):
    """ Unit tests for Nary Join related rules"""
    @staticmethod
    def get_phys_plan_root(query, num_server, child_size=None):
        dlog = RACompiler()
        dlog.fromDatalog(query)
        dlog.optimize(myrialang.MyriaHyperCubeAlgebra(
            FakeCatalog(num_server, child_size)))
        ret = dlog.physicalplan
        assert isinstance(ret, algebra.Parallel)
        assert len(ret.children()) == 1
        ret = ret.children()[0]
        assert isinstance(ret, algebra.Store)
        return ret.input

    @SkipTest
    def test_merge_to_nary_join(self):
        """ Test the rule merging binary join to nary join.
        """
        def in_cond_idx(position, conditions):
            for i, cond in enumerate(conditions):
                if position in cond:
                    return i
            raise Exception("Cannot find attribute in join conditions")

        # 1. triangular join
        triangle_join = testNaryJoin.get_phys_plan_root(
            "A(x,y,z):-R(x,y),S(y,z),T(z,x)", 64)
        # test root operator type
        self.assertIsInstance(triangle_join, algebra.NaryJoin)
        # test arity of join conditions
        self.assertEqual(len(triangle_join.conditions), 3)
        # test join conditions
        conds = []
        for cond in triangle_join.conditions:
            conds.append([attribute.position for attribute in cond])
        self.assertEqual(in_cond_idx(0, conds), in_cond_idx(5, conds))
        self.assertEqual(in_cond_idx(1, conds), in_cond_idx(2, conds))
        self.assertEqual(in_cond_idx(3, conds), in_cond_idx(4, conds))

        # 2. star join
        star_join = testNaryJoin.get_phys_plan_root(
            "A(x,y,z,p):-R(x,y),S(x,z),T(x,p)", 64)
        # test root operator type
        self.assertIsInstance(star_join, algebra.NaryJoin)
        # test arity of join conditions
        self.assertEqual(len(star_join.conditions), 1)
        # test join conditions
        conds = []
        for cond in star_join.conditions:
            conds.append([attribute.position for attribute in cond])
        self.assertEqual(in_cond_idx(0, conds), in_cond_idx(2, conds))
        self.assertEqual(in_cond_idx(2, conds), in_cond_idx(4, conds))

    @SkipTest
    def test_hashed_column_mapping(self):
        """Test whether hashed columns are mapped to correct HC dimensions."""
        def get_hc_dim(expr, column):
            """Return the mapped dimension in hyper cube."""
            return expr.mapped_hc_dimensions[expr.hashed_columns.index(column)]

        def get_shuffle_producers(expr):
            """ Go two steps deeper in the tree, find shuffle producers.
            """
            ret = [child.input.input for child in expr.children()]
            return tuple(ret)

        # 1. triangular join
        triangular_join = testNaryJoin.get_phys_plan_root(
            "A(x,y,z):-R(x,y),S(y,z),T(z,x)", 64)
        shuffle_r, shuffle_s, shuffle_t = get_shuffle_producers(
            triangular_join)
        # x in R and x in T are shuffled to the same dimension
        self.assertEqual(get_hc_dim(shuffle_r, 0), get_hc_dim(shuffle_t, 1))
        # y in R and y in S are shuffled to the same dimension
        self.assertEqual(get_hc_dim(shuffle_r, 1), get_hc_dim(shuffle_s, 0))
        # z in S and z in T are shuffled to the same dimension
        self.assertEqual(get_hc_dim(shuffle_s, 1), get_hc_dim(shuffle_t, 0))

        # 2. star join
        star_join = testNaryJoin.get_phys_plan_root(
            "A(x,y,z,p):-R(x,y),S(x,z),T(x,p)", 64)
        shuffle_r, shuffle_s, shuffle_t = get_shuffle_producers(star_join)
        # x in R and x in S are shuffled to the same dimension
        self.assertEqual(get_hc_dim(shuffle_r, 0), get_hc_dim(shuffle_s, 0))
        # x in S and x in T are shuffled to the same dimension
        self.assertEqual(get_hc_dim(shuffle_s, 0), get_hc_dim(shuffle_t, 0))

    @SkipTest
    def test_cell_partition(self):
        def get_cell_partition(expr, dim_sizes, child_idx):
            children = expr.children()
            children = [c.input.input for c in children]
            child_schemes = [c.scheme() for c in children]
            conditions = myrialang.convert_nary_conditions(
                expr.conditions, child_schemes)
            return myrialang.HCShuffleBeforeNaryJoin.get_cell_partition(
                dim_sizes, conditions, child_schemes,
                child_idx, children[child_idx].hashed_columns)

        # 1. triangular join
        expr = testNaryJoin.get_phys_plan_root(
            "A(x,y,z):-R(x,y),S(y,z),T(z,x)", 64, {"R": 1, "S": 100, "T": 20})
        dim_sizes = [1, 2, 2]
        # test cell partition of scan r
        self.assertEqual(
            get_cell_partition(expr, dim_sizes, 0), [[0, 1], [2, 3]])
        # test cell partition of scan s
        self.assertEqual(
            get_cell_partition(expr, dim_sizes, 1), [[0], [1], [2], [3]])
        # test cell partition of scan t
        self.assertEqual(
            get_cell_partition(expr, dim_sizes, 2), [[0, 2], [1, 3]])

        # 2. chain join
        expr = testNaryJoin.get_phys_plan_root(
            "A(x,y,z,p):-R(x,y),S(y,z),T(z,p)", 64)
        dim_sizes = [2, 2]
        self.assertEqual(
            get_cell_partition(expr, dim_sizes, 0), [[0, 1], [2, 3]])
        self.assertEqual(
            get_cell_partition(expr, dim_sizes, 1), [[0], [1], [2], [3]])
        self.assertEqual(
            get_cell_partition(expr, dim_sizes, 2), [[0, 2], [1, 3]])

    @SkipTest
    def test_dim_size(self):
        def get_dim_size(expr):
            producer = expr.children()[0].input.input
            return producer.hyper_cube_dimensions

        def get_work_load(expr, dim_sizes):
            children = expr.children()
            children = [c.input for c in children]
            child_schemes = [c.scheme() for c in children]
            conditions = myrialang.convert_nary_conditions(
                expr.conditions, child_schemes)
            HSClass = myrialang.HCShuffleBeforeNaryJoin
            r_index = HSClass.reversed_index(child_schemes, conditions)
            child_sizes = [len(cs) for cs in child_schemes]
            return HSClass.workload(dim_sizes, child_sizes, r_index)

        # test triangle join with equal input size
        triangular_join = testNaryJoin.get_phys_plan_root(
            "A(x,y,z):-R(x,y),S(y,z),T(z,x)", 64)
        self.assertEqual(get_dim_size(triangular_join), (4, 4, 4))

        # test rectange join with equal input size
        rect_join = testNaryJoin.get_phys_plan_root(
            "A(x,y,z):-R(x,y),S(y,z),T(z,p), N(p,x)", 256)
        # note: there is more than one optimal [4,4,4,4] or [1,16,1,16] etc.
        self.assertEqual(get_work_load(rect_join, [4, 4, 4, 4]),
                         get_work_load(rect_join, get_dim_size(rect_join)))
