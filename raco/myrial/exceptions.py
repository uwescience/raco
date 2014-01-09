
class MyrialCompileException(Exception):
    pass

class MyrialUnexpectedEndOfFileException(MyrialCompileException):
    def __str__(self):
        return "Unexpected end-of-file"

class MyrialUnexpectedTokenException(MyrialCompileException):
    def __init__(self, token):
        self.token = token

    def __str__(self):
        return 'Unexpected token %s on line %d' % (self.token.value,
                                                   self.token.lineno)

class MyrialScanException(MyrialCompileException):
    def __init__(self, token):
        self.token = token

    def __str__(self):
        return 'Bad token string %s on line %d' % (self.token.value,
                                                   self.token.lineno)

class ColumnIndexOutOfBounds(Exception):
    pass
