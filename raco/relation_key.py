"""Representation of a Myria relation key.

Myria relations are identified by a tuple of user, program, relation_name."""


class RelationKey(object):
    def __init__(self, *args):
        if len(args) == 1:
            self.user = "public"
            self.program = "adhoc"
            self.relation = args[0]
        else:
            self.user, self.program, self.relation = args
        assert self.user and isinstance(self.user, basestring)
        assert self.program and isinstance(self.program, basestring)
        assert self.relation and isinstance(self.relation, basestring)

    def __repr__(self):
        return 'RelationKey(%r,%r,%r)' % (self.user, self.program,
                                          self.relation)

    def __str__(self):
        return '%s:%s:%s' % (self.user, self.program, self.relation)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(str(self))

    @classmethod
    def from_string(cls, s):
        """Create a RelationKey from a colon-delimited string."""
        toks = s.split(':')
        assert len(toks) <= 3

        return cls(*toks)
