from raco.tests import DatalogTest
import unittest

suite = unittest.TestLoader().loadTestsFromTestCase(DatalogTest)
unittest.TextTestRunner(verbosity=2).run(suite)
