"""Basic test of the command-line interface to Myrial."""

import subprocess
import unittest


class CliTest(unittest.TestCase):

    def test_cli(self):
        out = subprocess.check_output(['python', 'scripts/myrial',
                                       'examples/reachable.myl'])
        self.assertIn('DO', out)
        self.assertIn('WHILE', out)

    def test_cli_standalone_execute(self):
        out = subprocess.check_output(['python', 'scripts/myrial', '-f',
                                       'examples/standalone.myl'])
        self.assertIn('Dan Suciu,engineering', out)

    def test_cli_standalone_json(self):
        out = subprocess.check_output(['python', 'scripts/myrial', '-j',
                                       'examples/cast.myl'])
        self.assertIn('rawQuery', out)

    def test_cli_standalone_logical(self):
        out = subprocess.check_output(['python', 'scripts/myrial', '-l',
                                       'examples/standalone.myl'])
        self.assertIn("CrossProduct[FileScan", out)

    def test_cli_standalone_repr(self):
        out = subprocess.check_output(['python', 'scripts/myrial', '-r',
                                       'examples/standalone.myl'])
        self.assertIn("FileScan('./examples/dept.csv'", out)

    def test_cli_reserved_column_name(self):
        proc = subprocess.Popen(
            ['python', 'scripts/myrial', 'examples/bad_column_name.myl'],
            stdout=subprocess.PIPE)
        out = proc.communicate()[0]
        self.assertIn('The token "SafeDiv" on line 2 is reserved', out)
