from collections import Counter
import os
import re
from textwrap import dedent

import raco.algebra as algebra
from raco.expression import AttributeRef, toUnnamed
from raco.language.myrialang import convertcondition
import raco.types as types
from .flink_expression import FlinkExpressionCompiler

raco_to_type = {types.LONG_TYPE: "Long",
                types.INT_TYPE: "Integer",
                types.STRING_TYPE: "String",
                types.FLOAT_TYPE: "Float",
                types.DOUBLE_TYPE: "Double"}


def type_signature(scheme):
    n = len(scheme)
    t = [raco_to_type[t] for t in scheme.get_types()]
    return "Tuple{N}<{TYPES}>".format(N=n, TYPES=','.join(t))


def dataset_signature(scheme):
    return "DataSet<ts>".format(ts=type_signature(scheme))


def dot_types(scheme):
    t = ['{t}.class'.format(t=raco_to_type[t]) for t in scheme.get_types()]
    return ".types({})".format(','.join(t))


def compile_to_flink(raw_query, plan):
    flink = Flink(raw_query)
    flink.begin()
    list(plan.postorder(flink.visit))
    flink.end()
    return flink.get()


class Flink(algebra.OperatorCompileVisitor):
    """Produces Flink Java programs"""
    def __init__(self, query):
        self.lines = []
        assert isinstance(query, (str, unicode))
        self.query = query
        self.dataset_lines = {}
        self.operator_names = {}
        self.optype_names = Counter()
        self.indent = 0
        self.data_dir = "data"

    def alloc_operator_name(self, op):
        if isinstance(op, algebra.Scan):
            return op.relation_key.relation

        opname = op.opname()
        self.optype_names[opname] += 1
        return '{}{}'.format(opname, self.optype_names[opname])

    def dataset_path(self, dataset):
        return os.path.join(self.data_dir, dataset)

    @staticmethod
    def dataset_name(dataset):
        # replace all illegal chars
        return re.sub('[^0-9a-zA-Z_]', '_', dataset)

    def _load_dataset(self, dataset, scheme):
        assert isinstance(dataset, basestring) and len(dataset) > 0
        dataset = Flink.dataset_name(dataset)
        if dataset in self.dataset_lines:
            return False
        method = """
  private static DataSet<{ts}> load_{ds}(ExecutionEnvironment env) {{
    return env.readCsvFile("{base}/{ds}"){dt};
  }}""".format(ts=type_signature(scheme), ds=dataset,
               base='file:///tmp/flink', dt=dot_types(scheme))
        self.dataset_lines[dataset] = method
        return True

    def begin(self):
        query = '\n'.join('//   {line}'.format(line=line)
                          for line in dedent(self.query).strip().split('\n'))
        preamble = """
import org.apache.flink.api.java.*;
import org.apache.flink.api.java.tuple.*;
import org.apache.flink.api.common.functions.*;

// Original query:
{query}

public class FlinkQuery {{

  public static void main(String[] args) throws Exception {{

    final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
""".format(query=query).strip()  # noqa
        self.lines.append(preamble)
        self.indent = 2

    def _add_line(self, line):
        self.lines.append('{ind}{line}'.format(ind='  ' * self.indent,
                                               line=line))

    def _add_lines(self, lines):
        assert isinstance(lines, list)
        self.lines += ['{ind}{line}'.format(ind='  ' * self.indent,
                                            line=line)
                       for line in lines]

    def _add_op_comment(self, op):
        child_str = ','.join(self.operator_names[str(c)]
                             for c in op.children())
        if child_str:
            child_str = '[{cs}]'.format(cs=child_str)
        self._add_line('')
        self._add_line('// {op}{cs}'.format(op=op.shortStr(), cs=child_str))

    def _add_op_code(self, op, op_code, add_dot_types=True):
        op_str = str(op)
        if op_str in self.operator_names:
            self._add_line('// skipped -- already computed')
            return
        name = self.alloc_operator_name(op)
        self.operator_names[op_str] = name
        scheme = op.scheme()
        type_sig = type_signature(scheme)
        if add_dot_types:
            dt = dot_types(scheme)
        else:
            dt = ""
        self._add_line("DataSet<{ts}> {name} = {code}{dt};"
                       .format(ts=type_sig, name=name, code=op_code, dt=dt))

    def every(self, op):
        self._add_op_comment(op)

    def v_scan(self, scan):
        name = scan.relation_key.relation
        self._load_dataset(name, scan.scheme())
        self._add_op_code(scan,
                          "load_{ds}(env)".format(ds=Flink.dataset_name(name)),
                          add_dot_types=False)

    def v_store(self, store):
        name = store.relation_key.relation
        in_name = self.operator_names[str(store.input)]
        self._add_line('{inp}.writeAsCsv("{base}/{out}");'
                       .format(inp=in_name,
                               base='file:///tmp/flink',
                               out=name))

    def v_select(self, select):
        child = select.input
        child_sch = child.scheme()
        child_sig = type_signature(child_sch)

        cond = FlinkExpressionCompiler(child_sch).visit(select.condition)

        ff = """
FilterFunction<{cs}>() {{
    @Override
    public boolean filter({cs} t) {{
        return {cond};
    }}
}}""".format(cs=child_sig, cond=cond).strip()

        child_str = self.operator_names[str(select.input)]
        op_code = "{child}.filter(new {ff})".format(child=child_str, ff=ff)
        self._add_op_code(select, op_code, add_dot_types=False)

    def visit_column_select(self, apply):
        scheme = apply.scheme()
        cols = [str(toUnnamed(ref[1], scheme).position)
                for ref in apply.emitters]
        cols_str = ','.join(cols)
        in_name = self.operator_names[str(apply.input)]
        self._add_op_code(apply, "{inp}.project({cols})"
                          .format(inp=in_name, cols=cols_str))

    def v_apply(self, apply):
        # For now, only handle column selection
        if all(isinstance(e[1], AttributeRef) for e in apply.emitters):
            self.visit_column_select(apply)
            return

        raise NotImplementedError('v_apply of {}'.format(apply))

    def v_projectingjoin(self, join):
        scheme = join.scheme()

        # First we need the condition
        left_len = len(join.left.scheme())
        condition = convertcondition(join.condition, left_len, scheme)
        where_clause = (".where({lc}).equalTo({rc})"
                        .format(lc=','.join(str(c) for c in condition[0]),
                                rc=','.join(str(c) for c in condition[1])))

        # Now we need the project clause
        output_cols = [toUnnamed(ref, scheme).position
                       for ref in join.output_columns]
        for (i, c) in enumerate(output_cols):
            if c < any(output_cols[:i]):
                raise NotImplementedError("ProjectingJoin with unordered cols")

        left_cols = [i for i in output_cols if i < left_len]
        right_cols = [i - left_len for i in output_cols if i >= left_len]

        lc = ','.join(str(c) for c in left_cols)
        if left_cols:
            project_clause = ".projectFirst({lc})".format(lc=lc)
        else:
            project_clause = ""

        rc = ','.join(str(c) for c in right_cols)
        if right_cols:
            project_clause = ("{pc}.projectSecond({rc})"
                              .format(pc=project_clause, rc=rc))

        op_code = ("{left}.joinWithHuge({right}){where}{proj}"
                   .format(left=self.operator_names[str(join.left)],
                           right=self.operator_names[str(join.right)],
                           where=where_clause, proj=project_clause))
        self._add_op_code(join, op_code)

    def end(self):
        # Should be at the end of main
        assert self.indent == 2
        # .. execute the query
        self._add_line('env.execute("MyriaL query");')
        # .. end main
        self.indent -= 1
        self._add_line("}")

        for dataset in self.dataset_lines:
            self.lines.append(self.dataset_lines[dataset])

        # end class
        self.indent -= 1
        self._add_line("}")

    def get(self):
        assert self.indent == 0
        return '\n'.join(self.lines)