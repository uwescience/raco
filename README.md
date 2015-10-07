Raco, the Relational Algebra COmpiler
=====================================

A pure Python compiler and optimization framework for relational algebra.

[![Build Status](https://travis-ci.org/uwescience/raco.png?branch=master)](https://travis-ci.org/uwescience/raco)
[![Coverage Status](https://coveralls.io/repos/uwescience/raco/badge.png)](https://coveralls.io/r/uwescience/raco)

Raco takes as input a number of source languages and has a growing number of output languages.

Source languages include:

* Datalog
* SQL (Subset)
* MyriaL, the language for the UW Myria project

Output languages include:

* Logical relational algebra (+ while loop)
* The Myria physical algebra (in JSON)
* C++ physical algebra, C++ source code
* [Grappa](http://grappa.io) (distributed C++) physical algebra, Grappa source code
* Pseudocode algebra
* Python physical algebra
* SPARQL (partial)

Users can of course author programs by directly instantiating one of the intermediate or output algebras as well as one of the source languages.


# Setup
Requires Python 2.7 or higher 2.x

For development use:

```bash
pip install -r requirements-dev.txt
python setup.py develop
```

For normal use:

```bash
python setup.py install
```


# Run tests

To execute the tests, run `nosetests` in the root directory of the repository. 

A few hints: to print test names use `-v`
```
nosetests -v
```
And fail on first error use `-x`
```
nosetests -x -v
```

See `nosetests -h` for more options or consult the [nose documentation](https://nose.readthedocs.org).


#### Requirements for C++ backend tests
- C++11 compiler, i.e. gcc-4.7 or later, clang-3.3 or later
- sqlite3

# Example

Raco is the compiler for Myria. To try parsing and understanding a program written in the Myria language, use the included `myrial` utility.

Note that the commands below run the `myrial` utility from the included `scripts` directory. However, the install command above will in fact install `myrial` in your `$PATH`.


### Parse a Myrial program
```bash
% python scripts/myrial -p examples/sigma-clipping-v0.myl
[('ASSIGN', 'Good', ('SCAN', 'public:adhoc:sc_points')), ('ASSIGN', 'N', ('TABLE', (<raco.myrial.emitarg.SingletonEmitArg object at 0x101c04fd0>,))), ('DOWHILE', [('ASSIGN', 'mean', ('BAGCOMP', [('Good', None)], None, (<raco.myrial.emitarg.SingletonEmitArg object at 0x101c1c450>,))), ('ASSIGN', 'std', ('BAGCOMP', [('Good', None)], None, (<raco.myrial.emitarg.SingletonEmitArg object at 0x101c1c4d0>,))), ('ASSIGN', 'NewBad', ('BAGCOMP', [('Good', None)], (ABS((Good.v - Unbox)) > (Unbox * Unbox)), (<raco.myrial.emitarg.FullWildcardEmitArg object at 0x101c1c410>,))), ('ASSIGN', 'Good', ('DIFF', ('ALIAS', 'Good'), ('ALIAS', 'NewBad'))), ('ASSIGN', 'continue', ('BAGCOMP', [('NewBad', None)], None, (<raco.myrial.emitarg.SingletonEmitArg object at 0x101c1c8d0>,)))], ('ALIAS', 'continue')), ('DUMP', 'Good')]
```

### Show the logical plan of a Myrial program

```bash
% python scripts/myrial -l examples/sigma-clipping-v0.myl
Sequence
    StoreTemp(Good)[Scan(public:adhoc:sc_points)]
    StoreTemp(N)[Apply(2=2)[SingletonRelation]]
    DoWhile
        Sequence
            StoreTemp(mean)[Apply(val=$0)[GroupBy(; AVERAGE(v))[ScanTemp(Good,[('v', 'float')])]]]
            StoreTemp(std)[Apply(val=$0)[GroupBy(; STDEV(v))[ScanTemp(Good,[('v', 'float')])]]]
            StoreTemp(NewBad)[Apply(v=$0)[Select((ABS(($0 - $1)) > ($2 * $3)))[CrossProduct[CrossProduct[CrossProduct[ScanTemp(Good,[('v', 'float')]), ScanTemp(mean,[('val', None)])], ScanTemp(N,[('2', <type 'int'>)])], ScanTemp(std,[('val', None)])]]]]
            StoreTemp(Good)[Difference[ScanTemp(Good,[('v', 'float')]), ScanTemp(NewBad,[('v', None)])]]
            StoreTemp(continue)[Apply(($0 > 0)=($0 > 0))[GroupBy(; COUNT($0))[ScanTemp(NewBad,[('v', None)])]]]
        ScanTemp(continue,[('($0 > 0)', None)])
    StoreTemp(__OUTPUT0__)[ScanTemp(Good,[('v', 'float')])]
```

### Show the Myria physical plan of a Myrial program

```bash
% python scripts/myrial examples/sigma-clipping-v0.myl 
Sequence
    StoreTemp(Good)[MyriaScan(public:adhoc:sc_points)]
    StoreTemp(N)[MyriaApply(2=2)[SingletonRelation]]
    DoWhile
        Sequence
            StoreTemp(mean)[MyriaApply(val=$0)[MyriaGroupBy(; AVERAGE(v))[MyriaCollectConsumer[MyriaCollectProducer(@None)[MyriaScanTemp(Good,[('v', 'float')])]]]]]
            StoreTemp(std)[MyriaApply(val=$0)[MyriaGroupBy(; STDEV(v))[MyriaCollectConsumer[MyriaCollectProducer(@None)[MyriaScanTemp(Good,[('v', 'float')])]]]]]
            StoreTemp(NewBad)[MyriaApply(v=$0)[MyriaSelect((ABS(($0 - $1)) > ($2 * $3)))[MyriaCrossProduct[MyriaCrossProduct[MyriaCrossProduct[MyriaScanTemp(Good,[('v', 'float')]), MyriaBroadcastConsumer[MyriaBroadcastProducer[MyriaScanTemp(mean,[('val', None)])]]], MyriaBroadcastConsumer[MyriaBroadcastProducer[MyriaScanTemp(N,[('2', <type 'int'>)])]]], MyriaBroadcastConsumer[MyriaBroadcastProducer[MyriaScanTemp(std,[('val', None)])]]]]]]
            StoreTemp(Good)[Difference[MyriaScanTemp(Good,[('v', 'float')]), MyriaScanTemp(NewBad,[('v', None)])]]
            StoreTemp(continue)[MyriaApply(($0 > 0)=($0 > 0))[MyriaGroupBy(; COUNT($0))[MyriaCollectConsumer[MyriaCollectProducer(@None)[MyriaScanTemp(NewBad,[('v', None)])]]]]]
        MyriaScanTemp(continue,[('($0 > 0)', None)])
    StoreTemp(__OUTPUT0__)[MyriaScanTemp(Good,[('v', 'float')])]
```

### Visualize a Myria plan as a graph
Pass the `-d` option to `scripts/myrial`. Output omitted for brevity.

### Output the Myria physical plan as json (what MyriaX understands)
Pass the `-j` option to `scripts/myrial`. Output omitted for brevity.

# C++ and Grappa output (Radish)
Raco has backend compilers that emit C++ and [Grappa](http://grappa.io) programs for query plans. To find out how it works in depth, read the [UW technical report](https://homes.cs.washington.edu/~bdmyers/papers/UW-CSE-14-10-01.pdf).

### Output C++ plan and source program
```
# generate the query and save to join.cpp
scripts/myrial --cpp examples/join.myl

# build
mv join.cpp c_test_environment/
cd c_test_environment; make join.exe

# run
c_test_environment/join.exe INPUT_FILE.csv
```

### Run the full MyriaL -> Grappa tests
The default tests (just running `nosetests`) include tests for translation from MyriaL to Grappa code but do no checking of whether the Grappa program correctly executes the query. To actually run the Grappa queries: 

1. `export RACO_HOME=/path/to/raco`
2. get Grappa https://github.com/uwsampa/grappa and follow installation instructions in its BUILD.md
3. `export GRAPPA_HOME=/path/to/grappa`
4. run tests: run this command from the `$RACO_HOME` directory
```bash
PYTHONPATH=c_test_environment RACO_GRAPPA_TESTS=1 python -m unittest grappalang_myrial_tests.MyriaLGrappaTest
```


### Output Radish (Grappa) plan and source program
```
scripts/myrial -c examples/join.myl
# the query implemented in Grappa is in join.cpp
```

# More Raco
using Raco, manipulating plans, adding optimizer rules, developing Raco
see [Raco in myria-docs](https://github.com/uwescience/myria-docs/blob/master/raco.markdown)

# Authors and contact information

Raco's authors include Bill Howe, Andrew Whitaker, Daniel Halperin, Brandon Myers and Dominik Moritz at the University of Washington. Contact us at <raco@cs.washington.edu>.
