import unittest
from raco import RACompiler
import raco.expression as e
import raco.expression.boolean


class ExpressionTest(unittest.TestCase):
    def test_postorder(self):
        expr1 = e.MINUS(e.MAX(e.NamedAttributeRef("salary")),
                        e.MIN(e.NamedAttributeRef("salary")))
        expr2 = e.PLUS(e.LOG(e.NamedAttributeRef("salary")),
                       e.ABS(e.NamedAttributeRef("salary")))

        def isAggregate(expr):
            return isinstance(expr, e.BuiltinAggregateExpression)

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
