
import raco.expression as sexpr

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

class SingletonEmitArg(EmitArg):
    """An emit arg that defines a single column.

    e.g.: [FROM Emp EMIT double_salary = salary * 2"""

    def __init__(self, column_name, sexpr):
        self.column_name = column_name
        self.sexpr = sexpr

    def expand(self, symbols):
        # TODO: choose a default column_name if not supplied.
        return [(self.column_name, self.sexpr)]

def expand_relation(relation_name, symbols):
    """Expand a given relation into a list of column mappings."""
    assert relation_name in symbols
    op = symbols[relation_name]
    scheme = op.scheme()

    colnames = [x[0] for x in iter(scheme)]
    return [(colname, sexpr.DottedAttributeRef(relation_name, colname))
            for colname in colnames]

class TableWildcardEmitArg(EmitArg):
    """An emit arg that refers to all columns in a table.

    e.g., [FROM emp, dept EMIT dept.*]"""

    def __init__(self, relation_name):
        self.relation_name = relation_name

    def expand(self, symbols):
        return expand_relation(self.relation_name, symbols)

class FullWildCardEmitArg(EmitArg):
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
