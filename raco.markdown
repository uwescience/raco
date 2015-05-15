# Raco query frontend and middleware

(This document is still a WIP)

[Raco](https://github.com/uwescience/raco) parses query languages, optimizes query plans, and compiles plans to representations accepted by query processing systems.

## Command line usage

`scripts/myrial` provides provides Raco functionality on the command line.

```bash
# help
scripts/myrial -h

# generate a logical plan for a MyriaL query in examples/
scripts/myrial -l examples/join.myl

# Raco requires a catalog for MyriaL queries. All of the example queries
# use a catalog given in examples/catalog.py. See examples/catalog.py and raco/catalog.py for formatting information.
# scripts/myrial automatically searches for a catalog.py in the same directory
# as the provided query. You can also provide a custom path.
scripts/myrial --catalog=examples/catalog.py -l examples/join.myl

# (in next release) you will also be able to specify a url of a json catalog
# or the url of a myria instance
TODO

```

## Rule-based optimization
The (non-experimental) optimization of query plans is done with a heuristic rule-based planner.
Raco provides many useful rules in `raco/rules.py`. `Rule` is the super class of all rules. 

## How optimization works
A physical algebra provides an implementation of `opt_rules`, which just returns an ordered list
of rules to apply. The optimizer applies each rule breadth first to the entire query plan tree, in the order specified by the list.
This algorithm is very simplistic, but it works out okay right now.

## How to add a rule
1. first, just check that the rule you need or something very close doesn't already exist in `raco/rules.py` or one of the languages in `raco/language/*.py`. If it is a generic rule and you find it in one of the languages, please [submit a pull request]( moving it to `raco/rules.py`https://github.com/uwescience/raco/compare).
2. If adding a rule, subclass `Rule` from `raco/rules.py`. You must implemented two methods: `_str_` and `fire`.
`fire` checks if the rule is applicable to the given tree. If not then it should return the tree itself. If the rule does apply then `fire` should return a transformed tree. It is okay to mutate the input tree and return it: most of Raco's rules are currently doing this instead of keeping the input immutable and copying the whole tree.
3. Go to your algebra (e.g., `MyriaLeftDeepJoinAlgebra` in `raco/language/myrialang.py`) and instantiate your rule somewhere in the list returned by `opt_rules`.
