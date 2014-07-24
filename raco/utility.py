def emit(*args):
    """Return blocks of code as a string."""
    return "\n".join([str(x) for x in args if len(str(x)) > 0])


def emitlist(argslist):
    """Return blocks of code as a string."""
    return "\n".join([str(x) for x in argslist if len(str(x)) > 0])


def str_list_inner(L):
    """Convert members of L to a comma-separated string.
    List brackets are not added."""
    return ", ".join([str(obj) for obj in L])


def str_list(L):
    """Convert L to a string using str rather than repr for list members."""
    return "[{inn}]".format(inn=str_list_inner(L))


class Printable(object):
    @classmethod
    def opname(cls):
        return str(cls.__name__)

    def __str__(self):
        return self.opname()
