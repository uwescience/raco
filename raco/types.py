
"""Names of primitive types understand by raco."""

# Map from myrial token name to "official" type name.
myrial_type_map = {
    "STRING": "STRING_TYPE",
    "INT": "LONG_TYPE",
    "FLOAT": "FLOAT_TYPE",
    "BOOLEAN": "BOOLEAN_TYPE"
}

# Map from python primitive types to "official" types names
python_type_map = {
    int: "LONG_TYPE",
    bool: "BOOLEAN_TYPE",
    float: "FLOAT_TYPE",
    str: "STRING_TYPE"
}

# The set of official type names
type_names = myrial_type_map.values()
