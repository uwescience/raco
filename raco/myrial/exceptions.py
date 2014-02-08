
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

    def __str__(self):
        return 'Duplicate function definition for %s on line %d' % (self.funcname,
                                                                    self.lineno)

class NoSuchFunctionException(MyrialCompileException):
    def __init__(self, funcname, lineno):
        self.funcname = funcname
        self.lineno = lineno

    def __str__(self):
        return 'No such function definition for %s on line %d' % (self.funcname,
                                                                  self.lineno)

class InvalidArgumentList(MyrialCompileException):
    def __init__(self, funcname, expected_args, lineno):
        self.funcname = funcname
        self.expected_args = expected_args
        self.lineno = lineno

    def __str__(self):
        return "Incorrect number of arguments for %s(%s) on line %d" % (
            self.funcname, ','.join(expected_args), lineno)

class UndefinedVariableException(MyrialCompileException):
    def __init__(self, funcname, var, lineno):
        self.funcname = funcname
        self.var = var
        self.lineno = lineno

    def __str__(self):
        return "Undefined variable %s in function %s at line %d" % (
            self.var, self.funcname, self.lineno)

class DuplicateVariableException(MyrialCompileException):
    def __init__(self, funcname, lineno):
        self.funcname = funcname
        self.lineno = lineno

    def __str__(self):
        return "Duplicately defined in function %s at line %d" % (
            self.funcname, self.lineno)

class BadApplyDefinitionException(MyrialCompileException):
    def __init__(self, funcname, lineno):
        self.funcname = funcname
        self.lineno = lineno

    def __str__(self):
        return "Bad apply definition for in function %s at line %d" % (
            self.funcname, self.lineno)

class UnnamedStateVariableException(MyrialCompileException):
    def __init__(self, funcname, lineno):
        self.funcname = funcname
        self.lineno = lineno

    def __str__(self):
        return "Unnamed state variable in function %s at line %d" % (
            self.funcname, self.lineno)

class IllegalWildcardException(MyrialCompileException):
    def __init__(self, funcname, lineno):
        self.funcname = funcname
        self.lineno = lineno

    def __str__(self):
        return "Illegal use of wildcard in function %s at line %d" % (
            self.funcname, self.lineno)

class ColumnIndexOutOfBounds(Exception):
    pass