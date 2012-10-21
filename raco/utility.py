"""
Return a block of code as a string.
"""
def emit(*args):
  return "\n".join(["%s" % x for x in args])

class Printable:
  def opname(self):
    return str(self.__class__.__name__)

  def __str__(self):
    return self.opname()


