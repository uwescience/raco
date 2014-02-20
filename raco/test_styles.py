'''
Code taken from:
https://github.com/ckan/ckan/blob/master/ckan/tests/test_coding_standards.py
which is published under the AGPL

The aim of these tests is to check and improve the coding standards.
Common issues are tested for here and tests fail if they are discovered in
files that are either new or were previously good. Bad files are
blacklisted to prevent them throwing errors in many cases because of the
number of affected files e.g. PEP8.  However if files start to pass a test
will fail and the file should be removed from the blacklist so that it will
then be kept clean in future.

The idea is to slowly improve the code quality without having files
deteriorating when they do reach the required standard.

Please do not add new files to the list as any new files should meet the
current coding standards.  Please add comments by files that fail if there
are legitimate reasons for the failure.
'''


import sys
import os
import re
import cStringIO

import pep8

file_path = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(file_path, '.'))


def process_directory(directory, ext='.py'):
    base_len = len(base_path) + 1
    for (dirpath, dirnames, filenames) in os.walk(directory):
        # ignore hidden files and dir
        filenames = [f for f in filenames if not f[0] == '.']
        dirnames[:] = [d for d in dirnames if not d[0] == '.']
        for name in filenames:
            if name.endswith(ext):
                path = os.path.join(dirpath, name)
                filename = path[base_len:]
                yield path, filename


def output_errors(filename, errors):
    out = ['']
    out.append('-' * len(filename))
    out.append(filename)
    out.append('-' * len(filename))
    for error in errors:
        out.append(error)
    return '\n'.join(out)


def show_fails(msg, errors):
    if errors:
        msg = ['\n%s' % msg]
        for error in errors:
            msg.append(errors[error])
        msg.append('\n\nFailing Files:\n==============')
        msg += sorted(errors)
        raise Exception('\n'.join(msg))


def show_passing(msg, errors):
    if errors:
        raise Exception('\n%s\n\n' % msg + '\n'.join(sorted(errors)))


class TestPep8(object):
    ''' Check that .py files are pep8 compliant '''

    # PEP8 File exceptions
    #
    # The following files have known PEP8 errors.  When the files get to a
    # point of not having any such errors they should be removed from this
    # list to prevent new errors being added to the file.

    PEP8_BLACKLIST_FILES = [
        'clang-v1.py',
        'grappalang.py',
        'clang.py',
        'clangcommon.py'
    ]
    fails = {}
    passes = []
    done = False

    @classmethod
    def setup(cls):
        if not cls.done:
            cls.process()
        cls.done = True

    @classmethod
    def process(cls):
        blacklist = cls.PEP8_BLACKLIST_FILES
        for path, filename in process_directory(base_path):
            errors = cls.find_pep8_errors(filename=path)
            if errors and not filename in blacklist:
                cls.fails[filename] = output_errors(filename, errors)
            elif not errors and filename in blacklist:
                cls.passes.append(filename)

    def test_pep8_fails(self):
        msg = 'The following files have pep8 issues that need resolving'
        msg += '\nThey need removing from the test blacklist'
        show_fails(msg, self.fails)

    def test_pep8_pass(self):
        msg = 'The following files passed pep8 but are blacklisted'
        show_passing(msg, self.passes)

    @staticmethod
    def find_pep8_errors(filename=None, lines=None):

        try:
            sys.stdout = cStringIO.StringIO()
            checker = pep8.Checker(filename=filename, lines=lines)
            checker.check_all()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        errors = []
        for line in output.split('\n'):
            parts = line.split(' ', 2)
            if len(parts) == 3:
                location, error, desc = parts
                line_no = location.split(':')[1]
                errors.append('%s ln:%s %s' % (error, line_no, desc))
        return errors


class TestImportStar(object):
    ''' Find files using from xxx import * '''

    # Import * file exceptions
    #
    # The following files contain one or more `from ... import *` lines
    # which should not be used in where possible.  If the files get
    # fixed they should be removed from this list.
    #
    # import * is bad for many reasons and should be avoided.

    IMPORT_STAR_BLACKLIST_FILES = [
        'expression/__init__.py',
        'expression/aggregate.py',
        'myrial/parser.py',
        'operator_test.py'
    ]
    fails = {}
    passes = []
    done = False

    @classmethod
    def setup(cls):
        if not cls.done:
            cls.process()
        cls.done = True

    @classmethod
    def process(cls):
        blacklist = cls.IMPORT_STAR_BLACKLIST_FILES
        re_import_star = re.compile(r'^\s*from\s+.*\simport\s+\*')
        for path, filename in process_directory(base_path):
            f = open(path, 'r')
            count = 1
            errors = []
            for line in f:
                if re_import_star.search(line):
                    errors.append('%s ln:%s import *\n\t%s'
                                  % (filename, count, line))
                count += 1
            if errors and not filename in blacklist:
                cls.fails[filename] = output_errors(filename, errors)
            elif not errors and filename in blacklist:
                cls.passes.append(filename)

    def test_import_good(self):
        msg = 'The following files passed import * rules'
        msg += '\nThey need removing from the test blacklist'
        show_passing(msg, self.passes)

    def test_import_bad(self):
        msg = ('The following files have import * issues that need resolving\n'
               '`from ... import *` lines which should not be used'
               ' where possible.')
        show_fails(msg, self.fails)
