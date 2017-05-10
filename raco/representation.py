

class RepresentationProperties(object):

    def __init__(
            self,
            hash_partitioned=frozenset(),
            sorted=None,
            grouped=None,
            broadcasted=False):
        """
        @param hash_partitioned: None or set of AttributeRefs in hash key
        @param sorted: None or list of (AttributeRefs, ASC/DESC) in sort order
        @param grouped: None or list of AttributeRefs to group by

        None means that no knowledge about the interesting property is
        known
        """

        # TODO: make it a set of sets, representing a conjunction of hashes
        # TODO: for example, after a HashJoin($1=$4) we know h($1) && h($4)
        # TODO:     which is not equivalent to h($1, $4). Currently can only
        # TODO:     represent conjunctions of size 1
        self.hash_partitioned = hash_partitioned
        self.broadcasted = broadcasted

        assert not (len(self.hash_partitioned) > 0 and self.broadcasted), \
            "inconsistent state: cannot be partitioned and broadcasted"

        if sorted is not None or grouped is not None:
            raise NotImplementedError("sorted and grouped not yet supported")

    def __str__(self):
        return "{clazz}(hash: {hash_attrs}, broadcasted: {b})".format(
            clazz=self.__class__.__name__,
            hash_attrs=self.hash_partitioned,
            b=self.broadcasted)

    def __repr__(self):
        return "{clazz}({hp!r}, {sort!r}, {grp!r}, {br!r})".format(
            clazz=self.__class__.__name__,
            hp=self.hash_partitioned,
            sort=None,
            grp=None,
            br=self.broadcasted
        )

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior
        (that returns the id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))
