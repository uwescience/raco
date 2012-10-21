

class Relation:
  def __init__(self, name, sch):
    self.name = name
    self.scheme = sch

class FileRelation(Relation):
  pass

class ASCIIFile(FileRelation):
  pass
