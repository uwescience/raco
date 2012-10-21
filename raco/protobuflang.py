import boolean
import rules
import algebra
from logical_ra_pb2 import *
from protobuf import text_format
from language import Language

class Protobuf(Language):
  @classmethod
  def boolean_combine(cls, args, operator="and"):
    opstr = " %s " % operator 
    conjunc = opstr.join(["%s" % cls.compile_boolean(arg) for arg in args])
    return "(%s)" % conjunc

  @classmethod
  def compile_attribute(cls, attr):
    if isinstance(attr, boolean.PositionReference):
      return attr.position
    elif isinstance(attr, boolean.Literal):
      return attr.value
    else:
      raise TypeError("Protobuf only supports positional references: %s" % attr)

class ProtobufOperator:
  language = Protobuf

class ProtobufScan(algebra.Scan, ProtobufOperator):
  def compileme(self, resultsym):
    scan = LogicalRaOperator()
    scan.type = LogicalRaOperator.SCAN
    scan.name = "%s" % resultsym
    scan.scan.relation = "R"

    return text_format.MessageToString(scan)

class ProtobufSelect(algebra.Select, ProtobufOperator):
  def compileme(self, resultsym, inputsym):
    select = LogicalRaOperator()
    select.type = LogicalRaOperator.SELECT
    select.name = "%s" % resultsym
    select.select.childName = inputsym
    select.select.condition = Protobuf.compile_boolean(self.condition)
    return text_format.MessageToString(select)

class ProtobufProject(algebra.Project, ProtobufOperator):
  def compileme(self, resultsym, inputsym):
    project = LogicalRaOperator()
    project.type = LogicalRaOperator.PROJECT
    project.name = "%s" % resultsym
    project.project.childName = inputsym
    posi = [Protobuf.unnamed(attref,self.scheme()) for attref in self.columnlist]
    cols = [Protobuf.compile_attribute(attref) for attref in posi]
    project.project.column.extend(cols)
    return text_format.MessageToString(project)

class ProtobufJoin(algebra.Join, ProtobufOperator):
  def compileme(self, resultsym, leftsym, rightsym):

    if not isinstance(self.condition,boolean.EQ):
      msg = "The Protobuf compiler can only handle equi-join conditions of a single attribute: %s" % self.condition
      raise ValueError(msg)

    join = LogicalRaOperator()
    join.type = LogicalRaOperator.JOIN
    join.name="%s" % resultsym
    join.join.leftChildName = leftsym
    join.join.rightChildName = rightsym
 
    condition = Protobuf.compile_boolean(self.condition)
    leftattribute = Protobuf.unnamed(self.condition.left,self.scheme()) 
    rightattribute = Protobuf.unnamed(self.condition.right,self.scheme()) 
    leftattribute = Protobuf.compile_attribute(leftattribute)
    rightattribute = Protobuf.compile_attribute(rightattribute)

    join.join.leftColumn.extend([leftattribute])
    join.join.rightColumn.extend([rightattribute])

    return text_format.MessageToString(join)

class ProtobufAlgebra:
  language = Protobuf

  operators = [
  ProtobufJoin,
  ProtobufSelect,
  ProtobufProject,
  ProtobufScan
]
  rules = [
  rules.CrossProduct2Join(),
  rules.OneToOne(algebra.Join,ProtobufJoin),
  rules.OneToOne(algebra.Select,ProtobufSelect),
  rules.OneToOne(algebra.Project,ProtobufProject),
  rules.OneToOne(algebra.Scan,ProtobufScan)
]
 
