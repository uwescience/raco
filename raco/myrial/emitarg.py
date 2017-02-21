
from raco.expression import DottedRef, UnnamedAttributeRef, NamedAttributeRef
from raco.myrial.exceptions import *


class EmitArg(object):
    """Defines a single entry in an emit argument list"""

    def expand(self, symbols):
        """Expand an EmitArg into a list of columns.

        symbols is a symbol table that maps relation names to operators
        that define their output.  The order of mappings follows the order
        of arguments in the EMIT clause.

        The output is a list of tuples of the form (name, expression).
        name is a string; expression is an instance of raco.expression.
        """
        raise NotImplementedError()

    def get_statemods(self):
        """Return a list of state variables associated with the emitarg.

        :return: A list of tuples of the form (name, init_expr, update_expr)
        """
        return []


def resolve_attribute_index(idx, symbols):
    """Resolve a column name given a positional index."""

    for op in symbols.values():
        scheme = op.scheme()
        if idx < len(scheme):
            return scheme.getName(idx)
        idx -= len(scheme)

    raise ColumnIndexOutOfBounds(str(idx))


def resolve_dotted_ref(sx, symbols):
    """Resolve a column name given an unbox expression.

    e.g. [FROM A EMIT A.some_column]
    """
    if isinstance(sx.field, basestring):
        return sx.field
    else:
        assert isinstance(sx.field, int)
        op = symbols[sx.table_alias]
        scheme = op.scheme()
        return scheme.getName(sx.field)


def get_column_name(name, sx, symbols):
    """Create a  column name; generate a name if none was provided.

    :param name: The name supplied by the user, or None if no name provided.
    :param sx: The Expression that defines the output
    :param symbols: A mapping from relation name to Operator instances
    """

    if name:
        return name

    if isinstance(sx, NamedAttributeRef):
        return sx.name
    elif isinstance(sx, UnnamedAttributeRef):
        return resolve_attribute_index(sx.position, symbols)
    elif isinstance(sx, DottedRef):
        return resolve_dotted_ref(sx, symbols)
    else:
        return name


class NaryEmitArg(EmitArg):
    """An emit arg that defines one or more columns."""

    def __init__(self, column_names, sexprs, statemods):
        assert column_names is None or len(column_names) == len(sexprs)
        assert len(sexprs) >= 1

        self.column_names = column_names
        self.sexprs = sexprs
        self.statemods = statemods

    def expand(self, symbols):
        names = self.column_names
        if not names:
            names = [None] * len(self.sexprs)

        return [(get_column_name(n, x, symbols), x)
                for n, x in zip(names, self.sexprs)]

    def get_statemods(self):
        return self.statemods

    def __repr__(self):
        return 'NaryEmitArg(%r,%r,%r)' % (
            self.column_names, self.sexprs, self.statemods)


def expand_relation(relation_name, symbols):
    """Expand a given relation into a list of column mappings."""
    if relation_name not in symbols:
        raise NoSuchRelationException(relation_name)

    op = symbols[relation_name]
    scheme = op.scheme()

    colnames = [x[0] for x in iter(scheme)]
    return [(colname, DottedRef(relation_name, colname))
            for colname in colnames]


class TableWildcardEmitArg(EmitArg):
    """An emit arg that refers to all columns in a table.

    e.g., [FROM emp, dept EMIT dept.*]"""

    def __init__(self, relation_name):
        self.relation_name = relation_name

    def expand(self, symbols):
        return expand_relation(self.relation_name, symbols)

    def __repr__(self):
        return 'TableWildcardEmitArg(%r)' % self.relation_name


class FullWildcardEmitArg(EmitArg):
    """Emit all columns from the input.

    This is basically 'select *'
    """

    def __init__(self):
        pass

    def expand(self, symbols):
        cols = []
        for relation_name in symbols:
            cols.extend(expand_relation(relation_name, symbols))
        return cols

    def __repr__(self):
        return 'FullWildcardEmitArg()'
