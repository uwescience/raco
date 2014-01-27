Raco, the Relational Algebra COmpiler
=====================================

A pure Python compiler and optimization framework for relational algebra.

Build status: ![build status from Travis CI](https://travis-ci.org/uwescience/datalogcompiler.png)

Source languages include:

* Datalog
* SQL (TODO)
* Myrial, the language for the UW Myria project

Output languages include:

* Logical relational algebra
* The Myria physical algebra.
* A pseudocode algebra
* A Python algebra
* A C algebra
* A Grappa algebra

Users can of course author programs by directly instantiating one of the intermediate or output algebras as well as one of the source languages.


# Setup
Requires Python 2.7 or higher 2.x

For development use:

    python setup.py develop

For normal use:

    python setup.py install



# Example

We are currently using Raco mostly for Myria. To try parsing and understanding a program written in the Myria language, use the included `myrial` utility.

Note that the commands below run the `myrial` utility from the included `scripts` directory. However, the install command above will in fact install `myrial` in your `$PATH`.


### Parse a Myrial program
```
% python scripts/myrial -p examples/sigma-clipping-v0.myl
[('ASSIGN', 'Good', ('SCAN', 'public:adhoc:sc_points')), ('ASSIGN', 'N', ('TABLE', (<raco.myrial.emitarg.SingletonEmitArg object at 0x101c04fd0>,))), ('DOWHILE', [('ASSIGN', 'mean', ('BAGCOMP', [('Good', None)], None, (<raco.myrial.emitarg.SingletonEmitArg object at 0x101c1c450>,))), ('ASSIGN', 'std', ('BAGCOMP', [('Good', None)], None, (<raco.myrial.emitarg.SingletonEmitArg object at 0x101c1c4d0>,))), ('ASSIGN', 'NewBad', ('BAGCOMP', [('Good', None)], (ABS((Good.v - Unbox)) > (Unbox * Unbox)), (<raco.myrial.emitarg.FullWildcardEmitArg object at 0x101c1c410>,))), ('ASSIGN', 'Good', ('DIFF', ('ALIAS', 'Good'), ('ALIAS', 'NewBad'))), ('ASSIGN', 'continue', ('BAGCOMP', [('NewBad', None)], None, (<raco.myrial.emitarg.SingletonEmitArg object at 0x101c1c8d0>,)))], ('ALIAS', 'continue')), ('DUMP', 'Good')]
```

### Show the logical plan of a Myrial program

```
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

```
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

# Authors and contract information

Raco's authors include Bill Howe, Andrew Whitaker, Daniel Halperin, and Brandon Myers at the University of Washington. Contact us at <raco@cs.washington.edu>.
