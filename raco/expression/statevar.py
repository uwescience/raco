import collections

# This type represents a state variable, as used by StatefulApply and UDAs
StateVar = collections.namedtuple(
    'StateVar', ['name', 'init_expr', 'update_expr'])
