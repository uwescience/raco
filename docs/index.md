---
layout: default
title: RACO
group: "docs"
weight: 4
section: 4
---

# Using Raco directly

This document explains usage of the Raco library. Most users of Myria will interact through [Myria python](http://myria.cs.washington.edu/docs/myria-python/), so who is this document for?

- Users of Myria that need to hack their query plan (**no need** to edit JSON query plans manually!)
- Developers hoping for an introduction to Raco
- Developers that plan to write a new back end for Raco


## Use Raco from the command line

`scripts/myrial` provides Raco functionality on the command line.

### Help

Print the help message

```bash
scripts/myrial -h
```

### Logical plan

Generate a logical plan for a MyriaL query in examples/

```bash
scripts/myrial -l examples/join.myl
```

### Physical plan

See the physical plan, MyriaX is the default algebra to use

```bash
scripts/myrial examples/join.myl
```

### Catalog

Raco requires a catalog for MyriaL queries. All of the example queries in `examples/`
use the catalog defined in `examples/catalog.py`. See `examples/catalog.py` and `raco/catalog.py` for formatting information.
`scripts/myrial` automatically searches for a `catalog.py` file in the same directory
as the provided query. You can also provide a custom path.
```bash
scripts/myrial --catalog=examples/catalog.py -l examples/join.myl
```

To connect to the real catalog of a Myria instance, see [using Raco from python](https://github.com/uwescience/raco/blob/master/docs/index.md#use-raco-from-python).

### Compiled plan

Get the JSON used to submit the query plan to MyriaX REST interface
```bash
scripts/myrial -j example/join.myl
```

### Intermediate plan files

There is also a python string representation of the query plan. This is valid
python code that you can give back to Raco.

```bash
# output the plan for the query (join.raco is just a file containing python code)
scripts/myrial -r example/join.myl >join.raco
cat join.raco
# run raco on the plan in join.raco instead of on a MyriaL query
scripts/myrial --plan join.raco
```

## Use Raco from python

You can write python scripts to construct query plans and compile them to back end systems like the Myria JSON format.

Below is the boilerplate template for going from MyriaL query to JSON query plan, using the catalog from [a running Myria instance](http://myria.cs.washington.edu/docs/). Fill in your Myria hostname and the query to translate.

```python
import raco.myrial.parser as parser
import raco.myrial.interpreter as interpreter
from raco.backends.myria.catalog import MyriaCatalog
from raco.backends.myria.connection import MyriaConnection

# connect to your Myria instance's catalog
connection = MyriaConnection(
   hostname=<url of your myria instance>,
   port=8753,
   ssl=False)
catalog = MyriaCatalog(connection)

_parser = parser.Parser()

statement_list = _parser.parse("""
<put myrial query here>
""")

processor = interpreter.StatementProcessor(catalog, True)
processor.evaluate(statement_list)

# here we print the logical, physical, and json versions of the plan for illustration purposes
print processor.get_logical_plan()
print processor.get_physical_plan()
print processor.get_json()
```

### Plan manipulation

Using Raco's python API, it is possible to manipulate the query plan at either
the logical or physical level.

#### Example (simple)

Often users of MyriaX want to partition a table. 

This is possible in MyriaL with Store:
```sql
T1 = scan(public:vulcan:edgesConnected);
store(T1, public:vulcan:edgesConnectedSort, [$0, $1, $3]);
```

In the past, MyriaL did not have this Shuffle syntax.
However we could still easily build the query plan we wanted. Here is an example of using MyriaX shuffle to partition a table in the MyMergerTree astronomy application. Don't feel daunted by the length of this example; most of the code is boilerplate to get the plan from a query. The action is at "this is the actual plan manipulation".

[vulcan.py catalog is here](https://gist.github.com/bmyerz/8fe4107eb8faff6221e8)

```python
from raco.catalog import FromFileCatalog
import raco.myrial.parser as parser
import raco.myrial.interpreter as interpreter
import raco.algebra as alg
from raco.expression.expression import UnnamedAttributeRef

# get the schema
catalog = FromFileCatalog.load_from_file("vulcan.py")
_parser = parser.Parser()

# We can have Raco start us with a plan that is close to the one we want by giving it a MyriaL query.
# Here we start with scan, store. We'll modify it to get scan, shuffle, store.
statement_list = _parser.parse("""
T1 = scan(public:vulcan:edgesConnected);
store(T1, public:vulcan:edgesConnectedSort);
""")
processor = interpreter.StatementProcessor(catalog, True)
processor.evaluate(statement_list)

# we will add the shuffle into the logical plan
p = processor.get_logical_plan()

# This is the actual plan manipulation; just insert a Shuffle. Since the
# operators are all unary (single-input) this just looks like linked-list insertion.
tail = p.args[0].input
p.args[0].input = alg.Shuffle(tail, [UnnamedAttributeRef(0), UnnamedAttributeRef(1), UnnamedAttributeRef(3)])
                                    # Shuffle columns

# output json query plan for MyriaX
p = processor.get_physical_plan()
p = processor.get_json()
print p
```

#### Example (a bit more complex)

Suppose Raco chooses to perform a join by shuffling both inputs.
However, we may know that the right input is much smaller and so we really
want to do a broadcast join.

```python
from raco.catalog import FromFileCatalog
import raco.myrial.parser as parser
import raco.myrial.interpreter as interpreter
import raco.backends.myria as alg
from raco.expression.expression import UnnamedAttributeRef

# get the schema
catalog = FromFileCatalog.load_from_file("vulcan.py")
_parser = parser.Parser()

# Get the default Raco plan for the join
statement_list = _parser.parse("""
T1 = scan(public:vulcan:edgesConnected);
s = select * from T1 a, T1 b where b.currentTime=0 and a.nextGroup=b.currentGroup;
store(s, public:vulcan:joined);
""")
processor = interpreter.StatementProcessor(catalog, True)
processor.evaluate(statement_list)

# will modify the physical plan, where a Join implementation is already chosen
p = processor.get_physical_plan()

# the plan is a symmetric hash join, shuffling both sides
print "before mod: ", p

# locate the MyriaSymmetricHashJoin operator
join = p.input.input.input
assert isinstance(join, alg.MyriaSymmetricHashJoin)

# modify the right side to replace shuffle with broadcast
rightChild = join.right.input.input
join.right = alg.MyriaBroadcastConsumer(alg.MyriaBroadcastProducer(rightChild))

# modify the left side to remove the shuffle
leftChild = join.left.input.input
join.left = leftChild

print "after mod: ", p

# output json query plan for MyriaX
p = processor.get_json()
print p
```

# Raco development

Here we provide information on extending Raco.

## Add a compiler backend

Raco currently emits code for MyriaX (+ SQL query push down into Postgres), Grappa/C++, and SQL databases. It has limited support for SciDB and SPARQL.

Every backend goes in its own directory within `raco/backends`. Here is an example:

```bash
raco/backends/myria/catalog.py: defines the Catalog interface for getting information about relations
raco/backends/myria/myria.py: defines the physical algebra for Myria and a list of optimization rules
raco/backends/myria/tests: tests specific to myria backend
raco/backends/myria/__init__.py: provides convenient import of public members using raco.backends.myria
```

`MyriaAlgebra` defines `opt_rules`, which is a list of optimization rules to apply in order.
`MyriaOperator` is the base class for operators in the physical algebra for Myria.

Compilation from a tree of `Operator`s to the target language can be implemented in any way you want.
For examples, see `MyriaOperator`'s `compileme` method and `GrappaOperator`'s `produce` and `consume` method.

## Add a new operator

Put your new operator for `<backend>` into `raco/backends/<backend>/<backend>.py`.
If you need to also add an operator to the logical operator, put it in `raco/algebra.py`.

## Rule-based optimization

The (non-experimental) optimization of query plans is done with a heuristic rule-based planner.
Raco provides many useful rules in `raco/rules.py`. `Rule` is the super class of all rules.

A physical algebra provides an implementation of `opt_rules`, which just returns an ordered list
of rules to apply. The optimizer applies each rule breadth first to the entire query plan tree, in the order specified by the list.
This algorithm is very simplistic, but it works out okay right now (see `raco/compile.py`).

### How to add a rule

1. first, just check that the rule you need or something very close doesn't already exist in `raco/rules.py` or one of the languages in `raco/language/*.py`. If it is a generic rule and you find it in one of the languages, please [submit a pull request]( moving it to `raco/rules.py`https://github.com/uwescience/raco/compare).
2. If adding a rule, subclass `Rule` from `raco/rules.py`. You must implement two methods: `_str_` and `fire`.
`fire` checks if the rule is applicable to the given tree. If not then it should return the tree itself. If the rule does apply then `fire` should return a transformed tree. It is okay to mutate the input tree and return it: most of Raco's rules are currently doing this instead of keeping the input immutable and copying the whole tree.
3. Go to your algebra (e.g., `MyriaLeftDeepJoinAlgebra` in `raco/backends/myria/myria.py`) and instantiate your rule somewhere in the list returned by `opt_rules`.


