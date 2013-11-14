
class MyrialCompileException(Exception):
    pass

class MyrialParseException(MyrialCompileException):
    pass

class MyrialScanException(MyrialCompileException):
    pass

class ColumnIndexOutOfBounds(Exception):
    pass
