

class Relation:
  def __init__(self, name, sch):
    self.name = name
    self.scheme = sch

  def __eq__(self, other):
    return self.name == other.name
    # and self.scheme == other.scheme


class FileRelation(Relation):
  pass

class ASCIIFile(FileRelation):
  pass
