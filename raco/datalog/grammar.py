"""
A parser for Purple programs.

The result is a parse object that can return a (recursive) relational algebra
expression.
"""

from pyparsing import Literal, CaselessLiteral, Word, delimitedList, \
    Optional, Combine, Group, alphas, nums, alphanums, oneOf, quotedString, \
    ZeroOrMore, restOfLine

import raco
from raco import expression
import raco.datalog.model as model


def show(x):
    print x
    return x

drop = lambda x: Literal(x).suppress()

# define Datalog tokens
ident = Word(alphas, alphanums + "_$")
predicate = ident.setName("Predicate")

E = CaselessLiteral("E")

# Get all the aggregate expression classes
aggregate_functions = raco.expression.aggregate_functions()

# All binary operators
binopclasses = expression.binary_ops()

# a list of string literals representing opcodes
opcodes = sum([oc.literals for oc in binopclasses], [])

binopstr = " ".join(opcodes)


def parsebinop(opexpr):
    "parse action for binary operators"
    left, opstr, right = opexpr

    for opclass in binopclasses:
        if opstr in opclass.literals:
            return opclass(left, right)

binop = oneOf(binopstr)
arithSign = Word("+-", exact=1)

realNum = Combine(Optional(arithSign) +
                  (Word(nums) + "." + Optional(Word(nums)) | ("." + Word(nums)))  # noqa
                  + Optional(E + Optional(arithSign) + Word(nums)))
realNum.setParseAction(lambda x: expression.NumericLiteral(float(x[0])))

intNum = Combine(Optional(arithSign) + Word(nums) +
                 Optional(E + Optional("+") + Word(nums)))
intNum.setParseAction(lambda x: expression.NumericLiteral(int(x[0])))

number = realNum | intNum

variable = ident.copy()
variable.setParseAction(lambda x: model.Var(x[0]))

quotedString.setParseAction(lambda x: expression.StringLiteral(x[0][1:-1]))

literal = quotedString | number

valueref = variable | literal


def mkterm(x):
    return model.Term(x)

term = (predicate
        + drop("(")
        + Group(delimitedList(valueref, ","))
        + drop(")")).setParseAction(mkterm)


def checkval(xs):
    left, op, right = xs[0]
    if op == '=':
        result = left == right
    else:
        result = eval(left + op + right)
    return result

groundcondition = Group(literal + binop + literal)
# groundcondition.setParseAction(checkval)

# TODO: deeper expression trees
condition = (valueref + binop + valueref)
condition.setParseAction(parsebinop)

body = delimitedList(term | groundcondition | condition, ",")
# .setParseAction(show) # lambda xs: [Term(x) for x in xs])

partitioner = drop("h(") + delimitedList(variable, ",") + drop(")")
partitioner.setParseAction(lambda x: model.PartitionBy(x))

allservers = Literal("*").setParseAction(lambda x: model.Broadcast())
server = drop("@") + (partitioner | allservers)
timeexpr = variable + oneOf("+ -") + Word(nums)
timeexpr.setParseAction(lambda xs: "".join([str(x) for x in xs]))

timestep = drop("#") + (intNum | timeexpr | variable)
timestep.setParseAction(lambda x: model.Timestep(x[0]))

# expressions without aggregates
# TODO more complete
simpleArithExpression = (valueref + binop + valueref)
simpleArithExpression.setParseAction(parsebinop)


def mkagg(x):
    opstr, arg = x
    for aggclass in aggregate_functions:
        if opstr.lower() == aggclass.__name__.lower():
            return aggclass(arg)
    raise "Aggregate Function %s not found among %s" % (opstr, aggregate_functions)  # noqa

aggregate = (Word(alphas) + drop("(") + variable + drop(")")) | \
            (Word(alphas) + drop("(") + simpleArithExpression + drop(")"))
aggregate.setParseAction(mkagg)


# expressions containing aggregates
# TODO deeper instead of enumeration
arithExpression = (aggregate + binop + aggregate) | \
                  (valueref + binop + valueref) | \
                  (aggregate + binop + valueref) | \
                  (valueref + binop + aggregate)
arithExpression.setParseAction(parsebinop)

# greedy parsing so put arithExpression first
headvalueref = arithExpression | aggregate | variable | literal

headterm = (predicate + Optional(server)
            + drop("(") + Group(delimitedList(headvalueref, ",")) + drop(")"))


def mkIDB(x):
    if len(x) == 4:
        idb = model.IDB(mkterm((x[0], x[2])), x[1], x[3])
    elif len(x) == 3:
        if isinstance(x[2], model.Timestep):
            idb = model.IDB(mkterm((x[0], x[1])), timestep=x[2])
        else:
            idb = model.IDB(mkterm((x[0], x[2])), x[1])
    else:
        idb = model.IDB(mkterm(x))
    return idb

head = (headterm + Optional(timestep) + drop(":-")).setParseAction(mkIDB)
# head.setParseAction(show)


def mkrule(x):
    """Workaround for AttributeError: Class Rule has no __call__ method when
    running through wsgi"""
    return model.Rule(x)

rule = (head + Group(body)
        + Optional(drop(";")) + Optional(drop(".")))
rule.setParseAction(mkrule)


def mkprogram(x):
    """Workaround for AttributeError: Class Rule has no __call__ method when
    running through wsgi"""
    return model.Program(x)

comment = (Literal("#") + restOfLine).suppress()

program = ZeroOrMore(rule | comment).setParseAction(mkprogram)


def parse(query):
    return program.parseString(query)[0]
