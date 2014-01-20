"""
Return blocks of code as a string.
"""
def emit(*args):
    return "\n".join([str(x) for x in args if len(str(x)) > 0])
  
"""
Return blocks of code as a string.
"""
def emitlist(argslist):
    return "\n".join([str(x) for x in argslist if len(str(x)) > 0])

class Printable(object):
    def opname(self):
        return str(self.__class__.__name__)

    @classmethod
    def opname(cls):
        return str(cls.__name__)

    def __str__(self):
        return self.opname()
