import unittest
from raco import RACompiler
import raco.expression as e
import raco.expression.boolean


def RATest(query):
    dlog = RACompiler()
    dlog.fromDatalog(query)
    # TODO: Testing for string equality, but we should do something like what
    # Andrew does -- evaluate the expressions on test data.
    return "%s" % dlog.logicalplan


class DatalogTest(unittest.TestCase):
    def test_join(self):
        join = """A(x,z) :- R(x,y), S(y,z)"""
        desiredresult = """[('A', Project($0,$3)[Join(($1 = $2))[Scan(public:adhoc:R), Scan(public:adhoc:S)]])]"""  # noqa
        testresult = RATest(join)
        self.assertEqual(testresult, desiredresult)

    def test_selfjoin(self):
        join = """A(x,z) :- R(x,y), R(y,z)"""
        desiredresult = """[('A', Project($0,$3)[Join(($1 = $2))[Scan(public:adhoc:R), Scan(public:adhoc:R)]])]"""  # noqa
        testresult = RATest(join)
        self.assertEqual(testresult, desiredresult)

    def test_triangle(self):
        join = """A(x,y,z) :- R(x,y), S(y,z), T(z,x)"""
        desiredresult = """[('A', Project($0,$1,$3)[Select(($3 = $4))[Join(($0 = $5))[Join(($1 = $2))[Scan(public:adhoc:R), Scan(public:adhoc:S)], Scan(public:adhoc:T)]]])]"""  # noqa
        testresult = RATest(join)
        self.assertEqual(testresult, desiredresult)

    def test_explicit_conditions(self):
        join = """A(x,y,z) :- R(x,y), S(w,z), x<y,y<z,y=w"""
        desiredresult = """[('A', Project($0,$1,$3)[Join((($1 < $3) and ($1 = $2)))[Select(($0 < $1))[Scan(public:adhoc:R)], Scan(public:adhoc:S)]])]"""  # noqa
        testresult = RATest(join)
        self.assertEqual(testresult, desiredresult)

    def test_select(self):
        select = "A(x) :- R(x,3)"
        desiredresult = """[('A', Project($0)[Select(($1 = 3))[Scan(public:adhoc:R)]])]"""  # noqa
        testresult = RATest(select)
        self.assertEqual(testresult, desiredresult)

    def test_select2(self):
        select = "A(x) :- R(x,y), S(y,z,4), z<3"
        desiredresult = """[('A', Project($0)[Join(($1 = $2))[Scan(public:adhoc:R), Select((($2 = 4) and ($1 < 3)))[Scan(public:adhoc:S)]]])]"""  # noqa
        testresult = RATest(select)
        self.assertEqual(testresult, desiredresult)

    def test_union(self):
        query = """
    A(x) :- B(x,y)
    A(x) :- C(y,x)
    """
        desiredresult = """[('A', Union[Project($0)[Scan(public:adhoc:B)], Project($1)[Scan(public:adhoc:C)]])]"""  # noqa
        testresult = RATest(query)
        self.assertEqual(testresult, desiredresult)

    def test_chained(self):
        query = """
    JustXBill(x) :- TwitterK(x,y)
    JustXBill2(x) :- JustXBill(x)
    JustXBillSquared(x) :- JustXBill(x), JustXBill2(x)
    """
        desiredresult = """[('JustXBillSquared', Project($0)[Join(($0 = $1))[Apply(x=$0)[Project($0)[Scan(public:adhoc:TwitterK)]], Apply(x=$0)[Project($0)[Apply(x=$0)[Project($0)[Scan(public:adhoc:TwitterK)]]]]]])]"""  # noqa
        testresult = RATest(query)
        self.assertEqual(testresult, desiredresult)

    def test_chained_rename(self):
        query = """
        A(x,z) :- R(x,y,z);
        B(w) :- A(3,w)
    """
        desiredresult = """[('B', Project($1)[Select(($0 = 3))[Apply(x=$0,w=$1)[Project($0,$2)[Scan(public:adhoc:R)]]]])]"""  # noqa
        testresult = RATest(query)
        self.assertEqual(testresult, desiredresult)

    def test_filter_expression(self):
        query = """filtered(src, dst, time) :- nccdc(src, dst, proto, time, a, b, c), time > 1366475761, time < 1366475821"""  # noqa
        desiredresult = "[('filtered', Project($0,$1,$3)[Select((($3 > 1366475761) and ($3 < 1366475821)))[Scan(public:adhoc:nccdc)]])]"  # noqa
        testresult = RATest(query)
        self.assertEquals(testresult, desiredresult)

    # thanks to #104
    def test_condition_flip(self):
        # this example is from sp2bench; a simpler one would be better
        query = """A(name1, name2) :- R(article1, 'rdf:type', 'bench:Article'),
                                      R(article2, 'rdf:type', 'bench:Article'),
                                      R(article1, 'dc:creator', author1),
                                      R(author1, 'foaf:name', name1),
                                      R(article2, 'dc:creator', author2),
                                      R(author2, 'foaf:name', name2),
                                      R(article1, 'swrc:journal', journal),
                                      R(article2, 'swrc:journal', journal),
                                      name1 < name2"""
        # don't check result, just don't have an error
        RATest(query)

    # test that attributes are correct amid multiple conditions
    def test_attributes_forward(self):
        query = "A(a) :- R(a,b), T(x,y,a,c), b=c"
        desiredresult = """[('A', Project($0)[Join((($0 = $4) and ($1 = $5)))[Scan(public:adhoc:R), Scan(public:adhoc:T)]])]"""  # noqa
        testresult = RATest(query)
        self.assertEquals(testresult, desiredresult)

    # test that attributes are correct amid multiple conditions and when the
    # order of variables in the terms is opposite of the explicit condition
    def test_attributes_reverse(self):
        query = "A(a) :- R(a,b), T(x,y,a,c), c=b"
        desiredresult = """[('A', Project($0)[Join((($0 = $4) and ($5 = $1)))[Scan(public:adhoc:R), Scan(public:adhoc:T)]])]"""  # noqa
        testresult = RATest(query)
        self.assertEquals(testresult, desiredresult)

    def test_apply_head(self):
        query = "A(a/b) :- R(a,b)"
        desiredresult = """[('A', Project($0)[Apply(_COLUMN0_=($0 / $1))[Scan(public:adhoc:R)]])]"""  # noqa
        testresult = RATest(query)
        self.assertEquals(testresult, desiredresult)

    def test_aggregate_head(self):
        query = "A(SUM(a)) :- R(a,b)"
        desiredresult = """[('A', Apply(_COLUMN0_=$0)[GroupBy(; SUM($0))[Scan(public:adhoc:R)]])]"""  # noqa
        testresult = RATest(query)
        self.assertEquals(testresult, desiredresult)

    def test_twoaggregate_head(self):
        query = "A(SUM(a),COUNT(b)) :- R(a,b)"
        desiredresult = """[('A', Apply(_COLUMN0_=$0,_COLUMN1_=$1)[GroupBy(; SUM($0),COUNT($1))[Scan(public:adhoc:R)]])]"""  # noqa
        testresult = RATest(query)
        self.assertEquals(testresult, desiredresult)

    def test_aggregate_head_group(self):
        query = "A(SUM(a),b) :- R(a,b)"
        desiredresult = """[('A', Apply(_COLUMN0_=$1,b=$0)[GroupBy($1; SUM($0))[Scan(public:adhoc:R)]])]"""  # noqa
        testresult = RATest(query)
        self.assertEquals(testresult, desiredresult)

    def test_aggregate_head_group_swap(self):
        query = "A(b,SUM(a)) :- R(a,b)"
        desiredresult = """[('A', Apply(b=$0,_COLUMN1_=$1)[GroupBy($1; SUM($0))[Scan(public:adhoc:R)]])]"""  # noqa
        testresult = RATest(query)
        self.assertEquals(testresult, desiredresult)

    def test_binop_aggregates(self):
        query = "A(SUM(b)+SUM(a)) :- R(a,b)"
        desiredresult = """[('A', Apply(_COLUMN0_=$0)[GroupBy(; (SUM($1) + SUM($0)))[Scan(public:adhoc:R)]])]"""  # noqa
        testresult = RATest(query)
        self.assertEquals(testresult, desiredresult)

    def test_aggregate_of_binop(self):
        query = "A(SUM(b+a)) :- R(a,b)"
        desiredresult = """[('A', Apply(_COLUMN0_=$0)[GroupBy(; SUM(($1 + $0)))[Scan(public:adhoc:R)]])]"""  # noqa
        testresult = RATest(query)
        self.assertEquals(testresult, desiredresult)

    def test_literal_expr(self):
        query = "A(a+1) :- R(a)"
        desiredresult = """[('A', Project($0)[Apply(_COLUMN0_=($0 + 1))[Scan(public:adhoc:R)]])]"""  # noqa
        testresult = RATest(query)
        self.assertEquals(testresult, desiredresult)


class ExpressionTest(unittest.TestCase):
    def test_postorder(self):
        expr1 = e.MINUS(e.MAX(e.NamedAttributeRef("salary")),
                        e.MIN(e.NamedAttributeRef("salary")))
        expr2 = e.PLUS(e.LOG(e.NamedAttributeRef("salary")),
                       e.ABS(e.NamedAttributeRef("salary")))

        def isAggregate(expr):
            return isinstance(expr, e.AggregateExpression)

        def classname(expr):
            return expr.__class__.__name__

        e1cls = [x for x in expr1.postorder(classname)]

        e2cls = [x for x in expr2.postorder(classname)]

        e1any = any(expr1.postorder(isAggregate))

        e2any = any(expr2.postorder(isAggregate))

        self.assertEqual(str(e1cls), """['NamedAttributeRef', 'MAX', 'NamedAttributeRef', 'MIN', 'MINUS']""")  # noqa
        self.assertEqual(str(e2cls), """['NamedAttributeRef', 'LOG', 'NamedAttributeRef', 'ABS', 'PLUS']""")  # noqa
        self.assertEqual(e1any, True)
        self.assertEqual(e2any, False)

    def test_visitor(self):
        class EvalVisitor(raco.expression.ExpressionVisitor):
            def __init__(self):
                self.stack = []

            def visit_NumericLiteral(self, numericLiteral):
                self.stack.append(numericLiteral.value)

            def visit_NEQ(self, binaryExpr):
                right = self.stack.pop()
                left = self.stack.pop()
                self.stack.append(left != right)

            def visit_AND(self, binaryExpr):
                right = self.stack.pop()
                left = self.stack.pop()
                self.stack.append(left and right)

            def visit_OR(self, binaryExpr):
                right = self.stack.pop()
                left = self.stack.pop()
                self.stack.append(left or right)

            def visit_NOT(self, unaryExpr):
                input = self.stack.pop()
                self.stack.append(not input)

            def visit_GTEQ(self, binaryExpr):
                pass

            def visit_UnnamedAttributeRef(self, unnamed):
                pass

            def visit_StringLiteral(self, stringLiteral):
                pass

            def visit_EQ(self, binaryExpr):
                pass

            def visit_GT(self, binaryExpr):
                pass

            def visit_LT(self, binaryExpr):
                pass

            def visit_LTEQ(self, binaryExpr):
                pass

            def visit_NamedAttributeRef(self, named):
                pass

            def visit_DIVIDE(self, binaryExpr):
                pass

            def visit_IDIVIDE(self, binaryExpr):
                pass

            def visit_MINUS(self, binaryExpr):
                pass

            def visit_NEG(self, binaryExpr):
                pass

            def visit_PLUS(self, binaryExpr):
                pass

            def visit_TIMES(self, binaryExpr):
                pass

        v = EvalVisitor()
        ex = e.AND(e.NEQ(e.NumericLiteral(1), e.NumericLiteral(2)),
                   e.NEQ(e.NumericLiteral(4), e.NumericLiteral(5)))
        ex.accept(v)
        self.assertEqual(v.stack.pop(), True)

        v = EvalVisitor()
        ex = e.AND(e.NEQ(e.NumericLiteral(1), e.NumericLiteral(2)),
                   e.NEQ(e.NumericLiteral(4), e.NumericLiteral(4)))
        ex.accept(v)
        self.assertEqual(v.stack.pop(), False)

        v = EvalVisitor()
        ex = e.AND(e.NEQ(e.NumericLiteral(1), e.NumericLiteral(2)),
                   e.NOT(e.NEQ(e.NumericLiteral(4), e.NumericLiteral(4))))
        ex.accept(v)
        self.assertEqual(v.stack.pop(), True)

        v = EvalVisitor()
        ex = e.NumericLiteral(0xC0FFEE)
        ex.accept(v)
        self.assertEqual(v.stack.pop(), 0xC0FFEE)
