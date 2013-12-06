"""Representation of a Myria relation key.

Myria relations are identified by a tuple of user, program, relation_name."""

class RelationKey(object):
    def __init__(self, user='public', program='adhoc', relation=None):
        assert relation
        self.user = user
        self.program = program
        self.relation = relation

    def __repr__(self):
        return 'RelationKey(%s,%s,%s)' % (self.user, self.program,
                                          self.relation)
    def __str__(self):
        return '%s:%s:%s' % (self.user, self.program, self.relation)

    def __eq__(self, other):
        return self.user == other.user and self.program == other.program \
            and self.relation == other.relation

    @classmethod
    def from_string(cls, s):
        """Create a RelationKey from a colon-delimited string."""
        toks = s.split(':')
        assert len(toks) <= 3

        args = {'relation' : toks[-1]}

        try:
            args['program'] = toks[-2]
            args['user'] = toks[-3]
        except IndexError:
            pass

        return cls(**args)
