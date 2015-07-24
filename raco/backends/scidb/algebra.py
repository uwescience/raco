
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

class GroupByToRegrid(rules.Rule):
    pass

class GroupByToRedimension(rules.Rule):
    pass

class SciDBAFLAlgebra(Algebra):
    """SciDB physical algebra"""
    def opt_rules(self, **kwargs):
        # replace logical operator with its corresponding SciDB operators
        scidbify = [
            rules.OneToOne(algebra.Store, SciDBStore),
            rules.OneToOne(algebra.Scan, SciDBScan),
            rules.OneToOne(algebra.UnionAll, SciDBConcat)
        ]
        all_rules = scidbify + [GroupByToRegrid, GroupByToRedimension]

        return list(itertools.chain(*all_rules))

    def __init__(self, catalog=None):
        self.catalog = catalog

HARDCODED_PLAN = "scan(SciDB__Demo__Waveform)"

def compile_to_afl(plan):
	#TODO this is wrong, obviously
	return HARDCODED_PLAN