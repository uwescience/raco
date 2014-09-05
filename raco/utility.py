import collections


def emit(*args):
    """Return blocks of code as a string."""
    return "\n".join([str(x) for x in args if len(str(x)) > 0])


def emitlist(argslist):
    """Return blocks of code as a string."""
    return "\n".join([str(x) for x in argslist if len(str(x)) > 0])


def real_str(obj, skip_out=False):
    """Convert the input object to a string, recursively stringifying elements
    inside of containers. If skip_out is True, the container bounds will not
    be displayed. E.g. real_str([1, 2]) == "[1,2]" but
    real_str([1, 2], skip_out=True) == "1,2"."""

    # Hack around basestrings being containers
    if (not isinstance(obj, basestring)
            and isinstance(obj, collections.Container)):

        if isinstance(obj, collections.Sequence):
            inner = ','.join(real_str(e) for e in obj)
            if skip_out:
                return inner
            return '[{inn}]'.format(inn=inner)
        elif isinstance(obj, collections.Mapping):
            inner = ','.join('{a}:{b}'.format(a=real_str(a), b=real_str(b))
                             for a, b in obj.items())
            if skip_out:
                return inner
            return '{{{inn}}}'.format(inn=inner)
        elif isinstance(obj, collections.Set):
            inner = ','.join(real_str(e) for e in obj)
            if skip_out:
                return inner
            return '{{{inn}}}'.format(inn=inner)
        else:
            raise NotImplementedError(
                "real_str(obj) for type(obj)={t}".format(t=type(obj)))

    return str(obj)


class Printable(object):
    @classmethod
    def opname(cls):
        return str(cls.__name__)

    def __str__(self):
        return self.opname()


# Optional raco dependency: termcolor
# Without it, coloring will not happen
def colored(s, color):
    return s
try:
    from termcolor import colored
except ImportError:
    pass
