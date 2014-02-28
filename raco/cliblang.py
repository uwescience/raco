from raco import algebra
from raco import expression
from raco import rules
from raco.language import Language


class CC(Language):
    @staticmethod
    def assignment(x, y):
        return "%s = %s;" % (x, y)

    @staticmethod
    def initialize(resultsym):
        return initialize % locals()  # TODO: is this ever used? # noqa

    @staticmethod
    def finalize(resultsym):
        return finalize % locals()  # TODO: is this ever used? # noqa

    @classmethod
    def boolean_combine(cls, args, operator="&&"):
        opstr = " %s " % operator
        conjunc = opstr.join(["(%s)" % cls.compile_boolean(arg)
                              for arg in args])
        return "( %s )" % conjunc

    """
    Expects unnamed perspective; use expression.to_unnamed_recursive e.g. to
    get there.
    """
    @staticmethod
    def compile_attribute(position):
        return 'tuple[%s]' % position


class CCOperator(object):
    language = CC


class FileScan(algebra.Scan, CCOperator):
    def compileme(self, resultsym):
        name = self.relation_key
        code = scan_template % locals()  # TODO: is this ever used? # noqa
        return code


class TwoPassSelect(algebra.Select, CCOperator):
    def compileme(self, resultsym, inputsym):
        pcondition = expression.to_unnamed_recursive(self.condition,
                                                     self.scheme())
        condition = CC.compile_boolean(pcondition)
        code = """

    bool condition_%(inputsym)s(const Tuple *tuple) {
      return %(condition)s
    }

    TwoPassSelect(&condition_%(inputsym)s, %(inputsym)s, %(resultsym)s);

    """ % locals()
        return code


class TwoPassHashJoin(algebra.Join, CCOperator):
    def compileme(self, resultsym, leftsym, rightsym):
        if len(self.attributes) > 1:
            raise ValueError("The C compiler can only handle equi-join conditions of a single attribute")  # noqa

        leftattribute, rightattribute = self.attributes[0]
        leftattribute = CC.compile_attribute(leftattribute)
        rightattribute = CC.compile_attribute(rightattribute)

        code = """

    HashJoin(&condition_%(inputsym)s, %(inputsym)s, %(resultsym)s);

    """ % locals()

        return code


class CCAlgebra(object):
    language = CC

    operators = [
        TwoPassHashJoin,
        TwoPassSelect,
        FileScan
    ]
    rules = [
        rules.OneToOne(algebra.Join, TwoPassHashJoin),
        rules.OneToOne(algebra.Select, TwoPassSelect),
        rules.OneToOne(algebra.Scan, FileScan)
    ]
