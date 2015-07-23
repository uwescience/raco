from raco.backends.myria.connection import MyriaConnection
from raco.backends.scidb.connection import SciDBConnection

class SciDBToMyria(object):
	source_type = SciDBConnection
	target_type = MyriaConnection

	def move(self, plan):
	    return self.import_(plan, self.export(plan))

	def import_(self, plan, metadata):
		pass

	def export(self, plan):
		return "foo"