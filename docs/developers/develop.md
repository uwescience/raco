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
