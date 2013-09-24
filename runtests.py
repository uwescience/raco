#!/usr/bin/env python
import unittest
import sys

loader = unittest.TestLoader()
# Find all modules that include test classes, we think
suite = loader.discover('raco', pattern='*test*.py')
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
sys.exit(not result.wasSuccessful())
