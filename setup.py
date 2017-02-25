from setuptools import setup

### Add find_packages function, see
# https://wiki.python.org/moin/Distutils/Cookbook/AutoPackageDiscovery
import os

def is_package(path):
    return (
        os.path.isdir(path) and
        os.path.isfile(os.path.join(path, '__init__.py'))
        )

def find_packages(path=".", base="", exclude=None):
    """Find all packages in path"""
    if not exclude:
        exclude = []
    packages = {}
    for item in os.listdir(path):
        dir = os.path.join(path, item)
        if is_package(dir) and dir not in exclude:
            if base:
                module_name = "{base}.{item}".format(base=base,item=item)
            else:
                module_name = item
            packages[module_name] = dir
            packages.update(find_packages(dir, module_name))
    return packages
###

setup(name='raco',
      version='1.3.3',
      description='Relational Algebra COmpiler',
      author='Bill Howe, Andrew Whitaker, Daniel Halperin',
      author_email='raco@cs.washington.edu',
      url='https://github.com/uwescience/raco',
      packages=find_packages(exclude=['clang']),
      package_data={'': ['c_templates/*.template','grappa_templates/*.template']},
      install_requires=['networkx', 'ply', 'pyparsing', 'SQLAlchemy', 'jinja2', 'requests', 'requests_toolbelt' ],
      scripts=['scripts/myrial']
      )
