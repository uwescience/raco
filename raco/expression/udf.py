import collections

# A user-defined function
Function = collections.namedtuple('Function', ['args', 'sexpr'])

# A user-defined stateful apply or UDA
StatefulFunc = collections.namedtuple(
    'StatefulFunc', ['args', 'statemods', "sexpr"])
