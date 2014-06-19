"""Convert myrial source programs to a plan representation."""

from raco.myrial.parser import Parser
from raco.myrial.interpreter import StatementProcessor


def myrial_to_ast(query, catalog=None):
    """Produce a myrial AST from a source program."""
    parser = Parser()
    processor = StatementProcessor(catalog)

    statements = parser.parse(query)
    return processor.evaluate(statements)


def myrial_to_plan(query, catalog=None, logical=False):
    parser = Parser()
    processor = StatementProcessor(catalog)

    statements = parser.parse(query)
    processor.evaluate(statements)
    if logical:
        return processor.get_logical_plan()
    else:
        return processor.get_physical_plan()


def myrial_to_logical_plan(query, catalog=None):
    return myrial_to_plan(query, catalog=catalog, logical=True)


def myrial_to_physical_plan(query, catalog=None):
    return myrial_to_plan(query, catalog=catalog, logical=False)
