import unittest
from raco.backends.myria.errors import MyriaError


class TestError(unittest.TestCase):
    def test_error(self):
        with self.assertRaises(MyriaError):
            raise MyriaError

if __name__ == '__main__':
    unittest.main()
