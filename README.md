Raco, the Relational Algebra COmpiler
=====================================

A pure Python compiler and optimization framework for relational algebra. Among other uses, [Raco is the query compiler for the Myria big data management system](https://github.com/uwescience/myria-stack).

[![Build Status](https://travis-ci.org/uwescience/raco.png?branch=master)](https://travis-ci.org/uwescience/raco)
[![Coverage Status](https://coveralls.io/repos/uwescience/raco/badge.png)](https://coveralls.io/r/uwescience/raco)

Raco takes as input a number of source languages and has a growing number of output languages.

Source languages include:

* MyriaL (includes SQL subset), the language for the UW Myria project
* Datalog

Output languages include:

* Logical relational algebra (+ while loop)
* The Myria physical algebra (in JSON)
* [Grappa](http://grappa.io) (distributed C++) source code programs
* C++ physical algebra, C++ source code programs
* Pseudocode algebra
* Python physical algebra
* Experimental: [SPARQL](https://www.w3.org/TR/rdf-sparql-query/), [SciDB](http://paradigm4.com/HTMLmanual/13.3/scidb_ug/ch01s04s01.html), [Spark dataframes](http://spark.apache.org/docs/latest/sql-programming-guide.html).

Users can of course author programs by directly instantiating one of the intermediate or output algebras as well as one of the source languages.


## Setup
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


## Run tests

Additional requirements for C++ back end tests

- C++11 compiler, i.e. gcc-4.7 or later, clang-3.3 or later
- sqlite3

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

## Example

Raco is the compiler for Myria. To try parsing and understanding a program written in the Myria language, use the included `myrial` utility.

Note that the commands below run the `myrial` utility from the included `scripts` directory. However, the install command above will in fact install `myrial` in your `$PATH`.



### Show the logical plan of a Myrial program

```bash
python scripts/myrial -l examples/sigma-clipping-v0.myl
```

```
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
python scripts/myrial examples/sigma-clipping-v0.myl
```

```
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

The `-d` option outputs a [dot file](www.graphviz.org/content/dot-language). The following command generates the plan for `join.myl` in a `png` image.

```bash
scripts/myrial -d examples/join.myl | dot -Tpng -o join.png
```

### Output the Myria physical plan as json

You can get the Myria physical plan as JSON, which you can give to Myria through its REST API.

```bash
scripts/myrial -j examples/select.myl
```

```
{"logicalRa": "MyriaStore(public:adhoc:OUTPUT)[MyriaSelect(($1 = 1))[MyriaScan(public:adhoc:employee)]]", "language": "myrial", "rawQuery": "Sequence[Store(public:adhoc:OUTPUT)[Select(($1 = 1))[Scan(public:adhoc:employee)]]]", "plan": {"fragments": [{"operators": [{"relationKey": {"userName": "public", "relationName": "employee", "programName": "adhoc"}, "opType": "TableScan", "opName": "MyriaScan(public:adhoc:employee)", "opId": 0}, {"opId": 1, "argPredicate": {"rootExpressionOperator": {"right": {"valueType": "LONG_TYPE", "type": "CONSTANT", "value": "1"}, "type": "EQ", "left": {"type": "VARIABLE", "columnIdx": 1}}}, "opType": "Filter", "opName": "MyriaSelect(($1 = 1))", "argChild": 0}, {"opType": "DbInsert", "argChild": 1, "argOverwriteTable": true, "relationKey": {"userName": "public", "relationName": "OUTPUT", "programName": "adhoc"}, "opName": "MyriaStore(public:adhoc:OUTPUT)", "opId": 2, "partitionFunction": null}]}], "type": "SubQuery"}}
```

### Only run the parser
```bash
python scripts/myrial -p examples/sigma-clipping-v0.myl
```

```
[('ASSIGN', 'Good', ('SCAN', 'public:adhoc:sc_points')), ('ASSIGN', 'N', ('TABLE', (<raco.myrial.emitarg.SingletonEmitArg object at 0x101c04fd0>,))), ('DOWHILE', [('ASSIGN', 'mean', ('BAGCOMP', [('Good', None)], None, (<raco.myrial.emitarg.SingletonEmitArg object at 0x101c1c450>,))), ('ASSIGN', 'std', ('BAGCOMP', [('Good', None)], None, (<raco.myrial.emitarg.SingletonEmitArg object at 0x101c1c4d0>,))), ('ASSIGN', 'NewBad', ('BAGCOMP', [('Good', None)], (ABS((Good.v - Unbox)) > (Unbox * Unbox)), (<raco.myrial.emitarg.FullWildcardEmitArg object at 0x101c1c410>,))), ('ASSIGN', 'Good', ('DIFF', ('ALIAS', 'Good'), ('ALIAS', 'NewBad'))), ('ASSIGN', 'continue', ('BAGCOMP', [('NewBad', None)], None, (<raco.myrial.emitarg.SingletonEmitArg object at 0x101c1c8d0>,)))], ('ALIAS', 'continue')), ('DUMP', 'Good')]
```

## Generate a C++ program

Raco has a backend compiler that emits C++.

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

## Generate a distributed C++/PGAS program

Raco has a back end compiler, Radish, that emits distributed C++ programs. In particular, Radish targets *partitioned global address space (PGAS)* languages, like [Grappa](http://grappa.io). Read [Compiling queries for high-performance computing](http://www.cs.washington.edu/tr/2016/02/UW-CSE-16-02-02.pdf) for more information on the internals of Radish.

### Generate a Grappa source program

```bash
scripts/myrial -c examples/join.myl
```

The query implemented in Grappa is now in `join.cpp`. To build and run the query, we recommend using [the Radish REST server](https://github.com/uwescience/radish-server).

### Run the full MyriaL-to-Grappa tests

The default tests (just running `nosetests`) include tests for *translation* from MyriaL to Grappa code but do no checking of whether the Grappa program correctly executes the query. To actually run the Grappa queries:

1. `export RACO_HOME=/path/to/raco`
2. get Grappa https://github.com/uwsampa/grappa and follow installation instructions in its BUILD.md
3. `export GRAPPA_HOME=/path/to/grappa`
4. run tests: run this command from the `$RACO_HOME` directory
```bash
PYTHONPATH=c_test_environment RACO_GRAPPA_TESTS=1 python -m unittest grappalang_myrial_tests.MyriaLGrappaTest
```

### More Radish examples

- [TPC-H benchmark](https://github.com/uwescience/tpch-radish)
- [Graph and matrix benchmarks](https://github.com/uwescience/sparseMatProjects/tree/master/myriaLQueries/radish)


## More in depth on Raco
To learn about calling Raco from python and manipulating plans, read [Using Raco directly](https://github.com/uwescience/raco/blob/master/docs/index.md)

To learn about developing Raco, read [developer doc](https://github.com/uwescience/raco/blob/master/docs/developers/develop.md).

## Authors and contact information

Raco's authors include Bill Howe, Andrew Whitaker, Daniel Halperin, Brandon Myers and Dominik Moritz at the University of Washington. Contact us at <raco@cs.washington.edu>.
