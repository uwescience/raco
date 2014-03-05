def emit(*args):
    """Return blocks of code as a string."""
    return "\n".join([str(x) for x in args if len(str(x)) > 0])


def emitlist(argslist):
    """Return blocks of code as a string."""
    return "\n".join([str(x) for x in argslist if len(str(x)) > 0])


class Printable(object):
    @classmethod
    def opname(cls):
        return str(cls.__name__)

    def __str__(self):
        return self.opname()
