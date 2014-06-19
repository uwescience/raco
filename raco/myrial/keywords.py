"""Emit all Myrial/SQL keywords as lowercase strings."""

from raco.myrial.scanner import (builtins, keywords,
                                 types, comprehension_keywords,
                                 word_operators)
from raco.expression.expressions_library import EXPRESSIONS


def get_keywords():
    """Return a list of Myrial/SQL keywords.

    This includes reserved lex tokens and system-defined functions.
    """
    return {
        'builtins': sorted(
            EXPRESSIONS.keys() + [kw.lower() for kw in builtins]),
        'keywords': sorted(kw.lower() for kw in keywords),
        'types': sorted(kw.lower() for kw in types),
        'comprehension_keywords': sorted(
            kw.lower() for kw in comprehension_keywords),
        'word_operators': sorted(kw.lower() for kw in word_operators),
    }
