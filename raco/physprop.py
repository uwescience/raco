
import utility


class ColumnEquivalenceClassSet(utility.CommonEqualityMixin):
    """A collection of column equivalence classes.

    A column equivalence class is a set of columns with the same value.  An
    equijoin is a common source of such columns.

    This class represents a collection of such equivalence classes.  By
    definition, the classes must not overlap.  Each classes has a
    representative, which is simply its smallest index.
    """

    def __init__(self, num_cols):
        """Instantiate a ColumnEquivalenceClassSet.

        Each column is placed in its own equivalence class.
        """

        rep_dict = {}  # map from representative to its members
        member_dict = {}  # map from a member to its representative

        # Initially, each column is its own equivalence class
        for i in range(num_cols):
            rep_dict[i] = {i}
            member_dict[i] = i

        self.rep_dict = rep_dict
        self.member_dict = member_dict

    def __str__(self):
        mems = ['%s : %s' % (k, v) for k, v in self.rep_dict.iteritems()]
        return '; '.join(mems)

    def merge(self, col1, col2):
        """Merge two equivalence classes."""

        _min = min(col1, col2)
        _max = max(col1, col2)
        assert _min != _max

        min_rep = self.member_dict[_min]
        max_rep = self.member_dict[_max]

        if min_rep == max_rep:
            return

        min_members = self.rep_dict[min_rep]
        max_members = self.rep_dict[max_rep]

        # Convert all member of the max's class to the new class
        min_members.update(max_members)

        for old_member in max_members:
            self.member_dict[old_member] = min_rep

        del self.rep_dict[max_rep]

    def get_equivalent_columns(self, col):
        rep = self.member_dict[col]
        return self.rep_dict[rep]

    def normalize(self, col_set):
        """Normalize a column set by replacing each member with an exemplar."""
        return set([self.member_dict[x] for x in col_set])


PARTITION_RANDOM = "RANDOM"
PARTITION_BROADCAST = "BROADCAST"
PARTITION_CENTRALIZED = "CENTRALIZED"


class PhysicalProperties(utility.CommonEqualityMixin):
    """Encodes various physical properties of data layout.

    This is subset of the SCOPE properties.  See "Incorporating Partitioning
    and Parallel Plans into the SCOPE Optimizer.
    """

    def __init__(self, part_info, cevs):
        """Instantiate a PhysicalProperties object.

        :param part_info: Partition information.  This is either a non-empty
        set of columns, or one of the special PARTION_* constants.
        :param cevs: An instance of ColumnEquivalenceClassSet
        """

        assert isinstance(cevs, ColumnEquivalenceClassSet)

        if isinstance(part_info, set):
            assert len(part_info) > 0
            self.part_info = cevs.normalize(part_info)
        else:
            self.part_info = part_info

        self.cevs = cevs

    def is_compatible(self, other):
        """Return True if other is compatible with self."""

        sp = self.part_info
        op = other.part_info
        if sp == op:
            return True
        if isinstance(sp, set) and isinstance(op, set):
            return op.issubset(sp)
        return False

    @classmethod
    def from_column_count(cls, part_info, num_cols):
        cls(part_info, ColumnEquivalenceClassSet(num_cols))
