"""Emit all Myrial/SQL keywords as lowercase strings."""

from raco.myrial.scanner import reserved
from raco.expression.expressions_library import EXPRESSIONS

def get_keywords():
    """Return a list of Myrial/SQL keywords.

    This includes reserved lex tokens and system-defined functions.
    """
    x = [res.lower() for res in reserved]
    x += EXPRESSIONS.keys()
    return x

if __name__ == '__main__':
    print get_keywords()
