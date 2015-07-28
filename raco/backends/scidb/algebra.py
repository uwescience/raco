
import logging
import itertools

from raco import rules
from raco.backends import Language, Algebra
from raco import algebra
from raco.expression import *


class SciDBLanguage(Language):
    pass


LOGGER = logging.getLogger(__name__)

class SciDBOperator(object):
  pass

class SciDBScan(algebra.Scan, SciDBOperator):
    def compileme(self):
        return "scan({})".format(self.name)


class SciDBStore(algebra.Store, SciDBOperator):
    def compileme(self, inputid):
        return "store({},{})".format(self.relation_key,inputid)


class SciDBConcat(algebra.UnionAll, SciDBOperator):
    def compileme(self, leftid, rightid):
        return "concat({},{})".format(leftid, rightid)

class SciDBRedimension(algebra.GroupBy, SciDBOperator):
    @staticmethod
    #TODO: Possible duplication of code.
    def agg_mapping(agg_expr):
        """Maps a BuiltinAggregateExpression to a SciDB string constant
        representing the corresponding aggregate operation."""
        #TODO: Supporting bare mininum expressions for the regrid in our query, needs to be made generic
        if isinstance(agg_expr, expression.BIN):
            return "BIN" + math.pow(2, agg_expr.n)
        elif isinstance(agg_expr, expression.SIGNED_COUNT):
            return "SIGNED_COUNT"
        elif isinstance(agg_expr, expression.AVG):
            return "AVG"
        raise NotImplementedError("SciDBRegrid.agg_mapping({})".format(
            type(agg_expr)))

    def compileme(self, inputid):
        group_fields = [ref for ref in self.grid_ids]
        #TODO: can be an aggregate on attributes or dimensions. Fix later to recognize this distinction
        built_ins = [agg_expr for agg_expr in self.aggregate_list
                     if isinstance(agg_expr, expression.BuiltinAggregateExpression)]

        aggregators =[]
        for i, agg_expr in enumerate(built_ins):
            aggregators.append("{}({})", SciDBRegrid.agg_mapping(agg_expr), group_fields[i])

        return "redimension({},{},{})".format(inputid, self.template_array, ",".join(aggregators))

    def shortStr(self):
        return super(SciDBRedimension, self).shortStr() + 'Parent Apply:' +self.parent_apply.shortStr()

class SciDBRegrid(algebra.GroupBy, SciDBOperator):
    @staticmethod
    def agg_mapping(agg_expr):
        """Maps a BuiltinAggregateExpression to a SciDB string constant
        representing the corresponding aggregate operation."""
        #TODO: Supporting bare mininum expressions for the regrid in our query, needs to be made generic
        if isinstance(agg_expr, expression.BIN):
            return "BIN" + math.pow(2, agg_expr.n)
        elif isinstance(agg_expr, expression.SIGNED_COUNT):
            return "SIGNED_COUNT"
        elif isinstance(agg_expr, expression.AVG):
            return "AVG"
        raise NotImplementedError("SciDBRegrid.agg_mapping({})".format(
            type(agg_expr)))


    def compileme(self, inputid):
        group_fields = [ref for ref in self.grid_ids]

        built_ins = [agg_expr for agg_expr in self.aggregate_list
                     if isinstance(agg_expr, expression.BuiltinAggregateExpression)]

        aggregators =[]
        for i, agg_expr in enumerate(built_ins):
            aggregators.append("{}({})", SciDBRegrid.agg_mapping(agg_expr), group_fields[i])

        #TODO: What about UDAs? Build support later on. Or since we are converting plans to scidb, is it necessary?
        return "regrid({},{},{})".format(inputid, ",".join(group_fields), ",".join(aggregators))

    def shortStr(self):
        return super(SciDBRegrid, self).shortStr() + 'Parent Apply:' +self.parent_apply.shortStr()

class SciDBAFLAlgebra(Algebra):

    """ SciDB algebra abstract class"""
    language = SciDBLanguage

    operators = [
        SciDBScan,
        SciDBStore,
        SciDBConcat,
        SciDBRegrid,
        SciDBRedimension
    ]
    """SciDB physical algebra"""
    def opt_rules(self, **kwargs):
        # replace logical operator with its corresponding SciDB operators
        scidbify = [
            rules.OneToOne(algebra.Store, SciDBStore),
            rules.OneToOne(algebra.Scan, SciDBScan),
            rules.OneToOne(algebra.UnionAll, SciDBConcat)
        ]
        all_rules = scidbify + [GroupByToRegridOrRedminension(), StoreToSciDBStore()]

        return all_rules

    def __init__(self, catalog=None):
        self.catalog = catalog

'''
class CountToDimensions(rules.Rule):
    pass
    # select max(i) from X => dimensions(X)[0]
'''
'''
class GroupByToRegrid(rules.Rule):
    def fire(self, expr):
        # if GroupBy
        # Look for dimension calculation:
        #   floor(dim1 / size1)
        #   (assumes that dim is 0:N)
        #   (assumes that dim is 0:N)
'''

class GroupByToRegridOrRedminension(rules.Rule):
    def fire(self, expr):
         # Look for a GroupBy-Apply pair.
        if isinstance(expr, algebra.Apply):
            # Check if the input to the operator is a groupby
            childop = expr.input
            if isinstance(childop, algebra.GroupBy):
                # Looking for dim / size operation.
                # Todo: Crazy amount of hardcoding. Fix Later

                is_regrid = False
                for gb_operation in childop.grouping_list:
                    if isinstance(gb_operation, CAST):
                        if isinstance(gb_operation.input, FLOOR):
                            if isinstance(gb_operation.input.input, DIVIDE):
                                if isinstance(gb_operation.input.input.right, NumericLiteral):
                                    is_regrid = True

                if is_regrid:
                    scidb_regridop = SciDBRegrid(childop.grouping_list, childop.aggregate_list, childop.input)
                    scidb_regridop.parent_apply = algebra.Apply()
                    scidb_regridop.parent_apply.copy(expr)
                    return scidb_regridop
                else:
                    scidb_redimension = SciDBRedimension(childop.grouping_list, childop.aggregate_list, childop.input)
                    scidb_redimension.parent_apply = algebra.Apply()
                    scidb_redimension.parent_apply.copy(expr)
                    return scidb_redimension
        return expr

    def __str__(self):
        return "GroupBy => ReGrid/ReDimension"

class StoreToSciDBStore(rules.Rule):
    def fire(self, expr):
        if isinstance(expr, algebra.Store) and not isinstance(expr, SciDBStore):
            scidb_store = SciDBStore(expr.relation_key, expr.plan)
            return scidb_store
        return expr

    def __str__(self):
        return "Store => SciDBStore"


def compile_to_afl(plan):
	#TODO Harcoded plan we wan't later we would want the actual conversion.
	ret = """
create temp array transform_1<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:127,256,0];
create temp array transform_2<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:63,256,0];
create temp array transform_3<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:31,256,0];
create temp array transform_4<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:15,256,0];
create temp array transform_5<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:7,256,0];
create temp array transform_6<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:3,256,0];
create temp array transform_7<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:1,256,0];
create temp array transform_8<value: double null, bucket:int64 null>[id=0:599,1,0, time=0:0,256,0];

create temp array out_transform_1<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_2<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_3<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_4<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_5<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_6<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_7<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];
create temp array out_transform_8<value:int64 null>[id=0:599,256,0, bucket=0:9,4294967296,0];

save(
  redimension(
    store(
      regrid(
        scan(input),
        1, 2,
        avg(value), bin1(value)),
        transform_1),
    out_transform_1,
    signed_count(bucket) as value),
  'socket://transform_1', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_1),
        1, 2,
        avg(value), bin2(value)),
        transform_2),
    out_transform_2,
    signed_count(bucket) as value),
  'socket://transform_2', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_2),
        1, 2,
        avg(value), bin4(value)),
        transform_3),
    out_transform_3,
    signed_count(bucket) as value),
  'socket://transform_3', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_3),
        1, 2,
        avg(value), bin8(value)),
        transform_4),
    out_transform_4,
    signed_count(bucket) as value),
  'socket://transform_4', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_4),
        1, 2,
        avg(value), bin16(value)),
        transform_5),
    out_transform_5,
    signed_count(bucket) as value),
  'socket://transform_5', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_5),
        1, 2,
        avg(value), bin32(value)),
        transform_6),
    out_transform_6,
    signed_count(bucket) as value),
  'socket://transform_6', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_6),
        1, 2,
        avg(value), bin64(value)),
        transform_7),
    out_transform_7,
    signed_count(bucket) as value),
  'socket://transform_7', -1, 'csv+');

save(
  redimension(
    store(
      regrid(
        scan(transform_7),
        1, 2,
        avg(value), bin128(value)),
        transform_8),
    out_transform_8,
    signed_count(bucket) as value),
  'socket://transform_8', -1, 'csv+');
	"""
	return ret