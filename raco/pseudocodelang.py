import algebra
import raco.rules
from language import Language


class PseudoCode(Language):
    @classmethod
    def new_relation_assignment(cls, rvar, val):
        return """
    %s
    %s
    """ % (cls.relation_decl(rvar), cls.assignment(rvar, val))

    @classmethod
    def relation_decl(cls, rvar):
        return ""

    @classmethod
    def assignment(cls, x, y):
        return "%s = %s" % (x, y)

    @staticmethod
    def comment(txt):
        return  "// %s" % txt

    @staticmethod
    def initialize(resultsym):
        return  """
    #include <relation.h>

    int tuplesize = 4*64;
    """

    @staticmethod
    def finalize(resultsym):
        return  """
    // write result somewhere?
    """

    @classmethod
    def boolean_combine(cls, args, operator="&&"):
        opstr = " %s " % operator
        conjunc = opstr.join(["(%s)" % cls.compile_boolean(arg) for arg in args])
        return "( %s )" % conjunc

    @staticmethod
    def compile_attribute(name):
        return 't->%s' % name


class PseudoCodeOperator:
    language = PseudoCode

class FileScan(algebra.Scan, PseudoCodeOperator):
    def compileme(self, resultsym):
        code = """
    // Build the input relation from disk, maybe?
    Relation *{resultsym} = new Relation();

    f = fopen({name});
    while not EOF:
      t = parsetuple(read(tuplesize),f)
      {resultsym}->insert(t);
    """.format(resultsym=resultsym, name=self.relation_key)
        return code

class TwoPassSelect(algebra.Select, PseudoCodeOperator):
    def compileme(self, resultsym, inputsym):
        condition = PseudoCode.compile_boolean(self.condition)
        code = """
    int size = 0;
    Tuple *t;
    for (int i=0; i<N; i++) {
      t = {inputsym}[i];
      if ({condition}) {
        size++
      }
    }

    {resultsym} = malloc(size*tuplesize);

    for (int i=0; i<N; i++) {
      t = {inputsym}[i];
      if ({condition}) {
        copy t to {resultsym};
      }
    }

    """.format(resultsym=resultsym, condition=condition)
        return code

class TwoPassHashJoin(algebra.Join, PseudoCodeOperator):
    def compileme(self, resultsym, leftsym, rightsym):
        code = """

    int size = 0;
    Tuple t;
    HashTable ht;
    for (int i=0; i<%(leftsym)s.size(); i++) {
      t = %(leftsym)s[i];
      ht.insert(t);
    }

    for (int i=0; i<%(rightsym)s.size(); i++) {
      t = %(rightsym)s[i];
      matches = probe ht with t;
      for each match:
        size++;
    }

    %(resultsym)s = malloc(tuplesize*size);

    for (int i=0; i<%(rightsym)s.size(); i++) {
      t = %(rightsym)s[i];
      matches = probe ht with t;
      for each match:
        copy t to %(resultsym)s;
    }


    """ % locals()
        return code


class PseudoCodeAlgebra:
    language = PseudoCode

    operators = [
    TwoPassHashJoin,
    TwoPassSelect,
    FileScan
  ]
    rules = [
    raco.rules.removeProject(),
    raco.rules.OneToOne(algebra.Join, TwoPassHashJoin),
    raco.rules.OneToOne(algebra.Select, TwoPassSelect),
    raco.rules.OneToOne(algebra.Scan, FileScan)
  ]
