import raco.myrial.parser as parser
import raco.myrial.interpreter as interpreter
import raco.compile


class ClangProcessor:
    def __init__(self, catalog):
        self.parser = parser.Parser()
        self.processor = interpreter.StatementProcessor(catalog)

    def get_plan(self, query, **kwargs):
        """Get the MyriaL query plan for a query"""
        statements = self.parser.parse(query)
        self.processor.evaluate(statements)
        if kwargs.get('logical', False):
            return self.processor.get_logical_plan(**kwargs)
        else:
            return self.processor.get_physical_plan(**kwargs)

    def get_physical_plan(self, query, **kwargs):
        """Get the physical plan for a MyriaL query"""
        kwargs['logical'] = False
        return self.get_plan(query, **kwargs)

    def write_source_code(self, query, basename, **kwargs):
        plan = self.get_physical_plan(query, kwargs)

        # generate code in the target language
        code = raco.compile.compile(plan)

        with open(basename+'.cpp', 'w') as f:
            f.write(code)
