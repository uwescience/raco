
"""Names of primitive types understand by raco.

Note that raco internally supports a limited set of types.  Different backends
can support a richer set of types, but these aren't understood by raco's
type system.  For example, raco doesn't distinguish between int and long,
so unsafe casts are not detected inside raco.
"""

# Internal and external types; these are understood by raco's type system
LONG_TYPE = "LONG_TYPE"
BOOLEAN_TYPE = "BOOLEAN_TYPE"
DOUBLE_TYPE = "DOUBLE_TYPE"
STRING_TYPE = "STRING_TYPE"
DATETIME_TYPE = "DATETIME_TYPE"
BLOB_TYPE = "BLOB_TYPE"

INTERNAL_TYPES = {LONG_TYPE, BOOLEAN_TYPE, DOUBLE_TYPE, STRING_TYPE, DATETIME_TYPE, BLOB_TYPE}  # noqa

# External only types; not understood by raco's type system
INT_TYPE = "INT_TYPE"
FLOAT_TYPE = "FLOAT_TYPE"

NUMERIC_TYPES = {LONG_TYPE, DOUBLE_TYPE}

TYPE_MAP = {k: k for k in INTERNAL_TYPES}
TYPE_MAP[INT_TYPE] = LONG_TYPE
TYPE_MAP[FLOAT_TYPE] = DOUBLE_TYPE
ALL_TYPES = TYPE_MAP.keys()


# Map from python primitive types to internal typess
python_type_map = {
    int: LONG_TYPE,
    bool: BOOLEAN_TYPE,
    float: DOUBLE_TYPE,
    str: STRING_TYPE
}

reverse_python_type_map = {v: k for k, v in python_type_map.iteritems()}


def map_type(s):
    """Convert an arbitrary type to an internal type."""
    return TYPE_MAP[s]


def parse_string(s, _type):
    """Convert from a string to an internal python representation."""
    assert _type in reverse_python_type_map
    return reverse_python_type_map[_type](s)
