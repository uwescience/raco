from myrialang import *
from raco import RACompiler
import unittest


class Catalog(object):
    def __init__(self, num_servers, child_sizes):
        self.num_servers = num_servers
        self.child_sizes = child_sizes

    def get_num_servers(self):
        return self.num_servers

    def get_child_sizes(self):
        return self.child_sizes


class testNaryJoin(unittest.TestCase):
    """ Unit tests for Nary Join related rules"""
    @staticmethod
    def get_phys_plan_root(query, num_server, child_size):
        dlog = RACompiler()
        dlog.fromDatalog(query)
        dlog.optimize(target=MyriaAlgebra(Catalog(num_server, child_size)),
                      multiway_join=True)
        return dlog.physicalplan[0][1]

    def test_merge_to_nary_join(self):
        """ Test the rule merging binary join to nary join.
        """
        def in_cond_idx(position, conditions):
            for i, cond in enumerate(conditions):
                if position in cond:
                    return i
            raise Exception("Cannot find attribute in join conditions")

        # 1. trianglular join
        triangle_join = testNaryJoin.get_phys_plan_root(
            "A(x,y,z):-R(x,y),S(y,z),T(z,x)", 64, [1, 1, 1])
        # test root operator type
        self.assertTrue(isinstance(triangle_join, algebra.NaryJoin))
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
            "A(x,y,z,p):-R(x,y),S(x,z),T(x,p)", 64, [1, 1, 1])
        # test root operator type
        self.assertTrue(isinstance(star_join, algebra.NaryJoin))
        # test arity of join conditions
        self.assertEqual(len(star_join.conditions), 1)
        # test join conditions
        conds = []
        for cond in star_join.conditions:
            conds.append([attribute.position for attribute in cond])
        self.assertEqual(in_cond_idx(0, conds), in_cond_idx(2, conds))
        self.assertEqual(in_cond_idx(2, conds), in_cond_idx(4, conds))

    def test_hashed_column_mapping(self):
        """Test whether hashed columns are mapped to correct HC dimensions."""
        def get_hc_dim(expr, column):
            """Return the mapped dimension in hyper cube."""
            return expr.mapped_hc_dimensions[expr.hashed_columns.index(column)]

        # 1. triangular join
        trianglular_join = testNaryJoin.get_phys_plan_root(
            "A(x,y,z):-R(x,y),S(y,z),T(z,x)", 64, [1, 1, 1])
        shuffle_r, shuffle_s, shuffle_t = trianglular_join.children()
        # x in R and x in T are shuffled to the same dimension
        self.assertEqual(get_hc_dim(shuffle_r, 0), get_hc_dim(shuffle_t, 1))
        # y in R and y in S are shuffled to the same dimension
        self.assertEqual(get_hc_dim(shuffle_r, 1), get_hc_dim(shuffle_s, 0))
        # z in S and z in T are shuffled to the same dimension
        self.assertEqual(get_hc_dim(shuffle_s, 1), get_hc_dim(shuffle_t, 0))

        # 2. star join
        star_join = testNaryJoin.get_phys_plan_root(
            "A(x,y,z,p):-R(x,y),S(x,z),T(x,p)", 64, [1, 1, 1])
        shuffle_r, shuffle_s, shuffle_t = tuple(star_join.children())
        # x in R and x in S are shuffled to the same dimension
        self.assertEqual(get_hc_dim(shuffle_r, 0), get_hc_dim(shuffle_s, 0))
        # x in S and x in T are shuffled to the same dimension
        self.assertEqual(get_hc_dim(shuffle_s, 0), get_hc_dim(shuffle_t, 0))

    def test_cell_partition(self):
        def get_cell_partiton(expr, dim_sizes, child_idx):
            children = expr.children()
            child_shemes = [c.scheme() for c in children]
            conditions = convert_nary_conditions(expr.conditions, child_shemes)
            return HCShuffleBeforeNaryJoin.get_cell_partition(
                dim_sizes, conditions, child_shemes,
                child_idx, children[child_idx].hashed_columns)

        # 1. triangular join
        expr = testNaryJoin.get_phys_plan_root(
            "A(x,y,z):-R(x,y),S(y,z),T(z,x)", 64, [1, 100, 20])
        dim_sizes = [1, 2, 2]
        # test cell partion of scan r
        self.assertEqual(
            get_cell_partiton(expr, dim_sizes, 0), [[0, 1], [2, 3]])
        # test cell partition of scan s
        self.assertEqual(
            get_cell_partiton(expr, dim_sizes, 1), [[0], [1], [2], [3]])
        # test cell partition of scan t
        self.assertEqual(
            get_cell_partiton(expr, dim_sizes, 2), [[0, 2], [1, 3]])

        # 2. chain join
        expr = testNaryJoin.get_phys_plan_root(
            "A(x,y,z,p):-R(x,y),S(y,z),T(z,p)", 64, [1, 1, 1])
        dim_sizes = [2, 2]
        self.assertEqual(
            get_cell_partiton(expr, dim_sizes, 0), [[0, 1], [2, 3]])
        self.assertEqual(
            get_cell_partiton(expr, dim_sizes, 1), [[0], [1], [2], [3]])
        self.assertEqual(
            get_cell_partiton(expr, dim_sizes, 2), [[0, 2], [1, 3]])

if __name__ == '__main__':
    unittest.main()
