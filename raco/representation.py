__author__ = 'brandon'


class RepresentationProperties(object):
    def __init__(self, hash_partitioned=set(), sorted=None, grouped=None):
        """
        @param hash_partitioned: None or list of AttributeRefs in hash key
        @param sorted: None or list of (AttributeRefs, ASC/DESC) in sort order
        @param grouped: None or list of AttributeRefs to group by

        None means that no knowledge about the interesting property is
        known
        """
        self.hash_partitioned = hash_partitioned

        if sorted is not None or grouped is not None:
            raise NotImplementedError("sorted and grouped not yet supported")