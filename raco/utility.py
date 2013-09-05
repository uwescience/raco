"""
Return a block of code as a string.
"""
def emit(*args):
  return "\n".join([str(x) for x in args if len(str(x)) > 0])

class Printable:
  def opname(self):
    return str(self.__class__.__name__)

  def __str__(self):
    return self.opname()


