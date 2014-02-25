"""Basic test of the command-line interface to Myrial."""

import subprocess
import unittest


class CliTest(unittest.TestCase):

    def test_cli(self):
        out = subprocess.check_output(['python', 'scripts/myrial',
                                       'examples/reachable.myl'])
        self.assertIn('DO', out)
        self.assertIn('WHILE', out)
