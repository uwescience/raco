from collections import Counter
import os
from textwrap import dedent

import raco.algebra as algebra
from raco.expression import AttributeRef, toUnnamed
from raco.language.myrialang import convertcondition
import raco.types as types

raco_to_type = {types.LONG_TYPE: "Long",
                types.INT_TYPE: "Integer",
                types.STRING_TYPE: "String",
                types.FLOAT_TYPE: "Float",
                types.DOUBLE_TYPE: "Double"}


def type_signature(scheme):
    n = len(scheme)
    t = [raco_to_type[t] for t in scheme.get_types()]
    return "DataSet<Tuple{N}<{TYPES}>>".format(N=n, TYPES=','.join(t))


def dot_types(scheme):
    t = ['{t}.class'.format(t=raco_to_type[t]) for t in scheme.get_types()]
    return ".types({})".format(','.join(t))


def compile_to_stratosphere(raw_query, plan):
    strat = Stratosphere(raw_query)
    strat.begin()
    list(plan.postorder(strat.visit))
    strat.end()
    return strat.get()


class Stratosphere(algebra.OperatorCompileVisitor):
    """Produces Stratosphere Java programs"""
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
        opname = op.opname()
        self.optype_names[opname] += 1
        return '{}{}'.format(opname, self.optype_names[opname])

    def dataset_path(self, dataset):
        return os.path.join(self.data_dir, dataset)

    def _load_dataset(self, dataset, scheme):
        assert isinstance(dataset, basestring)
        if dataset in self.dataset_lines:
            return False
        method = """
  private static {ts} load{ds}(ExecutionEnvironment env) {{
    return env.readCsvFile("{base}/{ds}"){dt};
  }}""".format(ts=type_signature(scheme), ds=dataset,
               base='file:///tmp/stratosphere', dt=dot_types(scheme))
        self.dataset_lines[dataset] = method
        self.lines.append('{ind}{ts} {ds} = load{ds}(env);'
                          .format(ind='  ' * self.indent,
                                  ts=type_signature(scheme),
                                  ds=dataset))
        return True

    def begin(self):
        comments = '\n'.join('// {line}'.format(line=line)
                             for line in dedent(self.query).strip().split('\n'))  # noqa
        preamble = """
import eu.stratosphere.api.java.*;
import eu.stratosphere.api.java.tuple.*;

{comments}

public class StratosphereQuery {{

  public static void main(String[] args) throws Exception {{

    final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
""".format(comments=comments)  # noqa
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

    def v_scan(self, scan):
        name = scan.relation_key.relation
        self._add_line('// {op}'.format(op=str(scan)))
        if self._load_dataset(name, scan.scheme()):
            self.operator_names[str(scan)] = name
        else:
            self._add_line('// skipped -- already loaded')

    def v_store(self, store):
        name = store.relation_key.relation
        in_name = self.operator_names[str(store.input)]
        self._add_line('// {op}'.format(op=store.shortStr()))
        self._add_line('{inp}.writeAsCsv("{base}/{out}");'
                       .format(inp=in_name,
                               base='file:///tmp/stratosphere',
                               out=name))

    def visit_column_select(self, apply):
        scheme = apply.scheme()
        cols = [str(toUnnamed(ref[1], scheme).position)
                for ref in apply.emitters]
        cols_str = ','.join(cols)
        in_name = self.operator_names[str(apply.input)]
        name = self.alloc_operator_name(apply)
        self.operator_names[str(apply)] = name
        self._add_line("// {op}".format(op=apply.shortStr()))
        self._add_line("{ts} {newop} = {inp}.project({cols}){dt};"
                       .format(ts=type_signature(scheme), newop=name,
                               inp=in_name, cols=cols_str,
                               dt=dot_types(scheme)))

    def v_apply(self, apply):
        # For now, only handle column selection
        if all(isinstance(e[1], AttributeRef) for e in apply.emitters):
            self.visit_column_select(apply)
            return

        raise NotImplementedError('v_apply of {}'.format(apply))

    def v_projectingjoin(self, join):
        scheme = join.scheme()
        left_name = self.operator_names[str(join.left)]
        right_name = self.operator_names[str(join.right)]
        name = self.alloc_operator_name(join)
        self.operator_names[str(join)] = name

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
            if i < any(output_cols[:c]):
                raise NotImplementedError("ProjectingJoin with unordered cols")

        left_cols = [i for i in output_cols if i < left_len]
        right_cols = [i - left_len for i in output_cols if i >= left_len]

        if len(left_cols) == 0 or len(right_cols) == 0:
            raise NotImplementedError("Stratosphere issue with semijoin: {}"
                                      .format(join.shortStr()))

        project_clause = (".projectFirst({lc}).projectSecond({rc})"
                          .format(lc=','.join(str(c) for c in left_cols),
                                  rc=','.join(str(c) for c in right_cols)))

        # Actually output the operator
        self._add_line("// {op}".format(op=join.shortStr()))
        self._add_line("{ts} {newop} = {left}.joinWithHuge({right}){where}{proj}{dt};"  # noqa
                       .format(ts=type_signature(scheme), newop=name,
                               left=left_name, right=right_name,
                               where=where_clause,
                               proj=project_clause,
                               dt=dot_types(scheme)))

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