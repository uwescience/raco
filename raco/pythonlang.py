import algebra
import raco.rules
from language import Language

class Python(Language):
    @classmethod
    def new_relation_assignment(cls, rvar, val):
        return """
    %s
    %s
    """ % (cls.relation_decl(rvar), cls.assignment(rvar, val))

    @classmethod
    def relation_decl(cls, rvar):
        # no type declarations necessary
        return "# decl %s" % rvar

    @staticmethod
    def assignment(x, y):
        return "%s = %s" % (x, y)

    @staticmethod
    def comment(txt):
        return  "# %s" % txt

    @staticmethod
    def initialize(resultsym):
        return  """
    import pyra
    import sampledb
    """

    @staticmethod
    def finalize(resultsym):
        return  """
    pyra.dump(%s)
    """ % resultsym

    @classmethod
    def boolean_combine(cls, args, operator="and"):
        opstr = " %s " % operator
        conjunc = opstr.join(["%s" % cls.compile_boolean(arg) for arg in args])
        return "(%s)" % conjunc

    @staticmethod
    def mklambda(body, var="t"):
        return ("lambda %s: " % var) + body

    @staticmethod
    def compile_attribute(name):
        return 't.%s' % name

class PythonOperator:
    language = Python

class pyScan(algebra.Scan, PythonOperator):
    def compileme(self, resultsym):
        opcode = """pyra.scan("%s", sampledb.__dict__)""" % self.relation_key
        code = self.language.new_relation_assignment(resultsym, opcode)
        return code

class pySelect(algebra.Select, PythonOperator):
    def compileme(self, resultsym, inputsym):
        opcode = """pyra.select(%s, %s)""" % (Python.mklambda(Python.compile_boolean(self.condition)), inputsym)
        code = self.language.new_relation_assignment(resultsym, opcode)
        return code

class pyHashJoin(algebra.Join, PythonOperator):
    def compileme(self, resultsym, leftsym, rightsym):
        opcode = """pyra.hashjoin(%s, %s, %s)\n""" % (self.language.compile_boolean(self.condition), leftsym, rightsym)
        code = self.language.new_relation_assignment(resultsym, opcode)
        return code

class PythonAlgebra:
    language = Python

    operators = [
    pyHashJoin,
    pySelect,
    pyScan
  ]
    rules = [
    raco.rules.removeProject(),
    raco.rules.OneToOne(algebra.Join, pyHashJoin),
    raco.rules.OneToOne(algebra.Select, pySelect),
    raco.rules.OneToOne(algebra.Scan, pyScan)
  ]
