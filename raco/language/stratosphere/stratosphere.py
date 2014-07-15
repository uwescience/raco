from collections import Counter
import os

import raco.algebra as algebra
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
            return
        method = """
  private static {ts} load{ds}(ExecutionEnvironment env) {{
    return env.readCsvFile("{base}/{ds}"){dt};
  }}""".format(ts=type_signature(scheme), ds=dataset,
               base='file:///tmp/stratosphere', dt=dot_types(scheme))
        self.dataset_lines[dataset] = method
        self.lines.append('{ind}{ts} {ds} = load{ds}(env);' 
                          .format(ind='  '*self.indent,
                                  ts=type_signature(scheme),
                                  ds=dataset))

    def begin(self):
        comments = '\n'.join('// {line}'.format(line=line)
                             for line in self.query.strip().split('\n'))
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
        self._load_dataset(name, scan.scheme())
        self.operator_names[str(scan)] = name

    def v_store(self, store):
        name = store.relation_key.relation
        in_name = self.operator_names[str(store.input)]
        self._add_line('// {op}'.format(op=str(store)))
        self._add_line('{inp}.writeAsCsv("{base}/{out}");'
                       .format(inp=in_name,
                               base='file:///tmp/stratosphere',
                               out=name))

    def v_unionall(self, store):
        name = store.relation_key.relation
        in_name = self.operator_names[str(store.input)]
        self._add_line('// {op}'.format(op=str(store)))
        self._add_line('{inp}.writeAsCsv("{base}/{out}");'
                       .format(inp=in_name,
                               base='file:///tmp/stratosphere', out=name))

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