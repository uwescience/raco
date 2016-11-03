# coding=utf-8
""" Exceptions that occur during Python->RACO conversion """


class PythonConvertException(Exception):
    """ Base class for conversion exceptions """
    pass


class PythonTokenException(PythonConvertException):
    """ Base exception class for errors associated with a specific token """
    def __init__(self, token, line, column):
        self.token = token
        self.line = line
        self.column = column


class PythonUnrecognizedTokenException(PythonTokenException):
    """ Error occurring when an unrecognized token is encountered """
    def __str__(self):
        return 'Conversion error at token %s on line %d:%d' % \
               (self.token, self.line, self.column)


class PythonOutOfRangeException(PythonUnrecognizedTokenException):
    """ Error occurring when a slice is out of range"""
    def __str__(self):
        return 'Slice out of range error near token %s on line %d:%d' % \
               (self.token, self.line, self.column)


class PythonSyntaxException(PythonConvertException):
    """ Error occurring when a Python source string contains a syntax error """
    def __init__(self, message, line, column):
        self.token = message
        self.line = line
        self.column = column

    def __str__(self):
        return 'Syntax error: %s (%d%s)' % \
               (self.token, self.line,
                ':' + str(self.column) if self.column else '')


class PythonUnsupportedOperationException(PythonSyntaxException):
    """ Error occurring when an unsupported operation is detected """
    def __str__(self):
        return 'Unsupported operation: %s (%d%s)' % \
               (self.token, self.line,
                ':' + str(self.column) if self.column else '')


class PythonArgumentException(PythonSyntaxException):
    """ Error occurring when a problem with an argument is detected """
    def __str__(self):
        return '%s (%d%s)' % \
               (self.token, self.line,
                ':' + str(self.column) if self.column else '')
