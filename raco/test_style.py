import subprocess
import sys
import unittest


class StyleTest(unittest.TestCase):
    "run flake8 with the right arguments and ensure all files pass"
    def test_style(self):
        try:
            subprocess.check_output(['flake8', '--ignore=F', 'raco'],
                                    stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print >> sys.stderr, e.output
            raise
