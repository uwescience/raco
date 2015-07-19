
import logging

from raco import rules
from raco.backends import Language, Algebra
from raco import algebra

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

class SciDBAFLAlgebra(Algebra):
    """ SciDB algebra abstract class"""
    language = SciDBLanguage

    operators = [
        SciDBScan,
        SciDBStore,
        SciDBConcat
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

class SciDBAFLAlgebra(Algebra):
    """SciDB physical algebra"""
    def opt_rules(self, **kwargs):
        # replace logical operator with its corresponding SciDB operators
        scidbify = [
            rules.OneToOne(algebra.Store, SciDBStore),
            rules.OneToOne(algebra.Scan, SciDBScan),
            rules.OneToOne(algebra.UnionAll, SciDBConcat),
        ]
        return scidbify

    def __init__(self, catalog=None):
        self.catalog = catalog
