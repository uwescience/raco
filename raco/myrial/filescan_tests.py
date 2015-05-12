
import collections

import raco.algebra
import raco.scheme as scheme
import raco.myrial.myrial_test as myrial_test
from raco import types


class FileScanTest(myrial_test.MyrialTestCase):

    def test_filescan(self):
        query = """
x = load("examples/load_options.csv",
csv(
    schema(column0:int, column1:string, column2:string, column3:float),
    delimiter="|", quote="~", escape="%", skip=2));
store(x, OUTPUT);
        """
        expected = collections.Counter([
            (1, "foo", "abc|def", 1.0),
            (2, "bar", "ghi|jkl", 2.0),
        ])
        self.check_result(query, expected)
