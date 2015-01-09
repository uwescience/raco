
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
        return 'Duplicate function definition for %s on line %d' % (self.funcname,  # noqa
                                                                    self.lineno)  # noqa


class NoSuchFunctionException(MyrialCompileException):
    def __init__(self, funcname, lineno):
        self.funcname = funcname
        self.lineno = lineno

    def __str__(self):
        return 'No such function definition for %s on line %d' % (self.funcname,  # noqa
                                                                  self.lineno)  # noqa


class ReservedTokenException(MyrialCompileException):
    def __init__(self, token, lineno):
        self.token = token
        self.lineno = lineno

    def __str__(self):
        return 'The token "%s" on line %d is reserved.' % (self.token,
            self.lineno)  # noqa


class InvalidArgumentList(MyrialCompileException):
    def __init__(self, funcname, expected_args, lineno):
        self.funcname = funcname
        self.expected_args = expected_args
        self.lineno = lineno

    def __str__(self):
        return "Incorrect number of arguments for %s(%s) on line %d" % (
            self.funcname, ','.join(self.expected_args), self.lineno)


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


class NestedTupleExpressionException(MyrialCompileException):
    def __init__(self, lineno):
        self.lineno = lineno

    def __str__(self):
        return "Illegal use of tuple expression on line %d" % self.lineno


class InvalidEmitList(MyrialCompileException):
    def __init__(self, function, lineno):
        self.function = function
        self.lineno = lineno

    def __str__(self):
        return "Wrong number of emit arguments in %s at line %d" % (
            self.function, self.lineno)


class IllegalColumnNamesException(MyrialCompileException):
    def __init__(self, lineno):
        self.lineno = lineno

    def __str__(self):
        return "Invalid column names on line %d" % self.lineno


class ColumnIndexOutOfBounds(Exception):
    pass


class SchemaMismatchException(MyrialCompileException):
    def __init__(self, op_name):
        self.op_name = op_name

    def __str__(self):
        return "Incompatible input schemas for %s operation" % self.op_name


class NoSuchRelationException(MyrialCompileException):
    def __init__(self, relname):
        self.relname = relname

    def __str__(self):
        return "No such relation: %s" % self.relname
