
from raco.backends.myria.connection import MyriaConnection
from raco.backends.myria.catalog import MyriaCatalog

import raco.myrial.interpreter as interpreter
import raco.myrial.parser as parser

from raco.backends.myria import MyriaLeftDeepTreeAlgebra

rest_url='https://rest.myria.cs.washington.edu:1776'
execution_url='https://myria-web.appspot.com'

connection = MyriaConnection(rest_url=rest_url, execution_url=execution_url)

catalog = MyriaCatalog(connection)

parser = parser.Parser()
processor = interpreter.StatementProcessor(catalog, True)

program = """
books = scan(Brandon:Demo:MoreBooks);
longerBooks = [from books where pages > 300 emit name];
store(longerBooks, Brandon:Demo:LongerBooks);
"""

statement_list = parser.parse(program)

processor.evaluate(statement_list)

pd = processor.get_physical_plan(target_alg=MyriaLeftDeepTreeAlgebra())

json = processor.get_json_from_physical_plan(pd)

result = connection.execute_query(json)

print result
