import collections

# A user-defined function
Function = collections.namedtuple('Function', ['args', 'sexpr'])

# A user-defined stateful apply
Apply = collections.namedtuple('Apply', ['args', 'statemods', "sexpr"])
