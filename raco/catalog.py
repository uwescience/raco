class Relation(object):
    def __init__(self, name, sch):
        self.name = name
        self._scheme = sch

    def __eq__(self, other):
        return self.name == other.name
        # and self.scheme == other.scheme

    def scheme(self):
        return self._scheme


class FileRelation(Relation):
    pass


class ASCIIFile(FileRelation):
    pass
