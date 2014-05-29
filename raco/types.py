
"""Names of primitive types understand by raco.

Note that raco internally supports a limited set of types.  Different backends
can support a richer set of types, but these aren't understood by raco's
type system.  For example, raco doesn't distinguish between int and long,
so unsafe casts are not detected inside raco.
"""

LONG_TYPE = "LONG_TYPE"
BOOLEAN_TYPE = "BOOLEAN_TYPE"
DOUBLE_TYPE = "DOUBLE_TYPE"
STRING_TYPE = "STRING_TYPE"
DATETIME_TYPE = "DATETIME_TYPE"

type_names = {LONG_TYPE, BOOLEAN_TYPE, DOUBLE_TYPE, STRING_TYPE, DATETIME_TYPE}


# Map from myrial token name to "official" type name.
myrial_type_map = {
    "STRING": STRING_TYPE,
    "INT": LONG_TYPE,
    "FLOAT": DOUBLE_TYPE,
    "BOOLEAN": BOOLEAN_TYPE
}

# Map from python primitive types to "official" types names
python_type_map = {
    int: LONG_TYPE,
    bool: BOOLEAN_TYPE,
    float: DOUBLE_TYPE,
    str: STRING_TYPE
}
