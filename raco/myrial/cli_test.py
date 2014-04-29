"""Basic test of the command-line interface to Myrial."""

import subprocess
import unittest


class CliTest(unittest.TestCase):

    def test_cli(self):
        out = subprocess.check_output(['python', 'scripts/myrial',
                                       'examples/reachable.myl'])
        self.assertIn('DO', out)
        self.assertIn('WHILE', out)

    def test_cli_reserved_column_name(self):
        proc = subprocess.Popen(
            ['python', 'scripts/myrial', 'examples/bad_column_name.myl'],
            stdout=subprocess.PIPE)
        out = proc.communicate()[0]
        self.assertIn('The token "SafeDiv" on line 2 is reserved', out)
