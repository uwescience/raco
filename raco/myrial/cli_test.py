"""Basic test of the command-line interface to Myrial."""

import subprocess
import unittest

class CliTest(unittest.TestCase):

    def test_cli(self):
        out = subprocess.check_output(['python', 'myrial.py',
                                       'examples/reachable.myl'])
        self.assertIn('DoWhile', out)
