'''
A parser for Datalog programs.

The result is a parse object that can return a (recursive) relational algebra expression.
'''

from pyparsing import Literal, CaselessLiteral, Word, Upcase, delimitedList, Optional, \
    Combine, Group, alphas, nums, alphanums, ParseException, Forward, oneOf, quotedString, \
    ZeroOrMore, restOfLine, Keyword

import raco
import model as model

def show(x):
  print x

# define Datalog tokens
ident = Word(alphas, alphanums + "_$")
predicate = ident.setName("Predicate")
drop = lambda x: Literal(x).suppress()

E = CaselessLiteral("E")

# Get all the binary operator classes
allclasses = [c for c in raco.boolean.__dict__.values() if not hasattr(c, "__class__")]
binopclasses = [opclass for opclass in allclasses
                   if issubclass(opclass,raco.boolean.BinaryBooleanOperator)
                   and opclass is not raco.boolean.BinaryBooleanOperator
                   and opclass is not raco.boolean.BinaryComparisonOperator]

# a list of string literals representing opcodes
opcodes = sum([oc.literals for oc in binopclasses], [])

binopstr = " ".join(opcodes)

# parse action for binary operators
def parsebinop(opexpr):
  left, opstr, right = opexpr
  
  for opclass in binopclasses:
    if opstr in opclass.literals:
      return opclass(left, right)
        
binop = oneOf(binopstr)
arithSign = Word("+-",exact=1)

realNum = Combine( Optional(arithSign) + ( Word( nums ) + "." + Optional( Word(nums) )  |
            ( "." + Word(nums) ) ) + 
            Optional( E + Optional(arithSign) + Word(nums) ) )
realNum.setParseAction(lambda x: raco.boolean.NumericLiteral(x[0])) 

intNum = Combine( Optional(arithSign) + Word( nums ) + 
            Optional( E + Optional("+") + Word(nums) ) )
intNum.setParseAction(lambda x: raco.boolean.NumericLiteral(x[0]))

number = realNum | intNum 

variable = ident.copy()
variable.setParseAction(lambda x: model.Var(x[0]))

quotedString.setParseAction(lambda x: raco.boolean.StringLiteral(x[0]))

literal = quotedString | number

valueref = variable | literal

def mkterm(x):
  return model.Term(x)

term = (predicate + drop("(") + Group(delimitedList(valueref, ",")) + drop(")")).setParseAction(mkterm)

def checkval(xs):
  left, op, right = xs[0]
  if op == '=':
    result = left == right
  else:
    result = eval(left + op + right)
  return result

groundcondition = Group(literal + binop + literal)
#groundcondition.setParseAction(checkval)

condition = (valueref + binop + valueref)
condition.setParseAction(parsebinop)

body = delimitedList(term | groundcondition | condition, ",")
#.setParseAction(show) #lambda xs: [Term(x) for x in xs])

head = (term + drop(":-")).setParseAction(lambda x: model.IDB(x[0]))

def mkrule(x):
  """Workaround for AttribtueError: Class Rule has no __call__ method when running through wsgi"""
  return model.Rule(x)

rule = (head + Group(body)).setParseAction(mkrule)

def mkprogram(x):
  """Workaround for AttribtueError: Class Rule has no __call__ method when running through wsgi"""
  return model.Program(x)

program = ZeroOrMore(rule).setParseAction(mkprogram)

def parse(query):
  return program.parseString(query)[0]
