import re

import raco.expression
import raco.types


def java_escape_str(s):
    # TODO this is not complete at all
    s = re.sub('[\\\\]', "\\\\", s)  # fix backslash
    s = re.sub('[\"]', '\\"', s)     # fix quote
    s = re.sub('[\n]', '\\n', s)     # fix newline
    return s


class FlinkExpressionCompiler(raco.expression.ExpressionVisitor):
    """Visit an expression, compile an equivalent Java string"""
    def __init__(self, scheme):
        self.scheme = scheme

    def get_type(self, expr):
        """Return the type of the expression given the input scheme"""
        return expr.typeof(self.scheme, None)

    def visit_AND(self, expr):
        return "({l}) && ({r})".format(l=self.visit(expr.left),
                                       r=self.visit(expr.right))

    def visit_OR(self, expr):
        return "({l}) || ({r})".format(l=self.visit(expr.left),
                                       r=self.visit(expr.right))

    def visit_NOT(self, expr):
        return "!({c})".format(c=self.visit(expr.input))

    def visit_NEG(self, expr):
        return "-({c})".format(c=self.visit(expr.input))

    def visit_PLUS(self, expr):
        return "({l}) + ({r})".format(l=self.visit(expr.left),
                                      r=self.visit(expr.right))

    def visit_MINUS(self, expr):
        return "({l}) - ({r})".format(l=self.visit(expr.left),
                                      r=self.visit(expr.right))

    def visit_TIMES(self, expr):
        return "({l}) * ({r})".format(l=self.visit(expr.left),
                                      r=self.visit(expr.right))

    def visit_DIVIDE(self, expr):
        return "({l}) * 1.0 / ({r})".format(l=self.visit(expr.left),
                                            r=self.visit(expr.right))

    def visit_IDIVIDE(self, expr):
        return "((long) (({l}) / ({r})))".format(l=self.visit(expr.left),
                                                 r=self.visit(expr.right))

    def visit_EQ(self, expr):
        l = self.visit(expr.left)
        r = self.visit(expr.right)
        if self.get_type(expr.left) == raco.types.STRING_TYPE:
            fmt = "({l}).equals({r})"
        else:
            fmt = "({l}) == ({r})"
        return fmt.format(l=l, r=r)

    def visit_NEQ(self, expr):
        l = self.visit(expr.left)
        r = self.visit(expr.right)
        if self.get_type(expr.left) == raco.types.STRING_TYPE:
            fmt = "!({l}).equals({r})"
        else:
            fmt = "({l}) != ({r})"
        return fmt.format(l=l, r=r)

    def visit_LT(self, expr):
        l = self.visit(expr.left)
        r = self.visit(expr.right)
        if self.get_type(expr.left) == raco.types.STRING_TYPE:
            fmt = "({l}).compareTo({r}) < 0"
        else:
            fmt = "({l}) < ({r})"
        return fmt.format(l=l, r=r)

    def visit_LTEQ(self, expr):
        l = self.visit(expr.left)
        r = self.visit(expr.right)
        if self.get_type(expr.left) == raco.types.STRING_TYPE:
            fmt = "({l}).compareTo({r}) <= 0"
        else:
            fmt = "({l}) <= ({r})"
        return fmt.format(l=l, r=r)

    def visit_GT(self, expr):
        l = self.visit(expr.left)
        r = self.visit(expr.right)
        if self.get_type(expr.left) == raco.types.STRING_TYPE:
            fmt = "({l}).compareTo({r}) > 0"
        else:
            fmt = "({l}) > ({r})"
        return fmt.format(l=l, r=r)

    def visit_GTEQ(self, expr):
        l = self.visit(expr.left)
        r = self.visit(expr.right)
        if self.get_type(expr.left) == raco.types.STRING_TYPE:
            fmt = "({l}).compareTo({r}) >= 0"
        else:
            fmt = "({l}) >= ({r})"
        return fmt.format(l=l, r=r)

    def visit_NamedAttributeRef(self, expr):
        pos = raco.expression.toUnnamed(expr, self.scheme).position
        return "t.f{pos}".format(pos=pos)

    def visit_UnnamedAttributeRef(self, expr):
        return "t.f{pos}".format(pos=expr.position)

    def visit_NumericLiteral(self, expr):
        t = self.get_type(expr)
        if t == raco.types.LONG_TYPE:
            return "{val}L".format(val=expr.value)
        elif t == raco.types.DOUBLE_TYPE:
            return "{val:g}".format(val=expr.value)
        else:
            raise NotImplementedError("Flink literal of type {t}".format(t=t))

    def visit_StringLiteral(self, expr):
        assert self.get_type(expr) == raco.types.STRING_TYPE
        return "\"{val}\"".format(val=java_escape_str(expr.value))