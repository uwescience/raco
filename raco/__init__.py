from raco.datalog.grammar import parse
from raco.compile import optimize

import logging
LOG = logging.getLogger(__name__)


class RACompiler(object):

    """Thin wrapper interface for lower level functions parse, optimize,
    compile"""

    def fromDatalog(self, program):
        """Parse datalog and convert to RA"""
        self.physicalplan = None
        self.source = program
        self.parsed = parse(program)
        LOG.debug("parser output: %s", self.parsed)
        self.logicalplan = self.parsed.toRA()

    def optimize(self, target, **kwargs):
        """Convert logical plan to physical plan"""
        self.physicalplan = optimize(self.logicalplan, target, **kwargs)
