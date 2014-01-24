
class MyrialCompileException(Exception):
    pass

class MyrialUnexpectedEndOfFileException(MyrialCompileException):
    def __str__(self):
        return "Unexpected end-of-file"

class MyrialParseException(MyrialCompileException):
    def __init__(self, token):
        self.token = token

    def __str__(self):
        return 'Parse error at token %s on line %d' % (self.token.value,
                                                       self.token.lineno)

class MyrialScanException(MyrialCompileException):
    def __init__(self, token):
        self.token = token

    def __str__(self):
        return 'Illegal token string %s on line %d' % (self.token.value,
                                                       self.token.lineno)

class DuplicateFunctionDefinitionException(MyrialCompileException):
    def __init__(self, funcname, lineno):
        self.funcname = funcname
        self.lineno = lineno

    def __str__self(self):
        return 'Duplicate function definition for %s on line %d' % (self.funcname,
                                                                    self.lineno)

class ColumnIndexOutOfBounds(Exception):
    pass
