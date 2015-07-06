{{comment}}
{{hashname}}.lookup_iter<&{{pipeline_sync}}>( {{keyval}}, [=]({{right_tuple_type}}& {{right_tuple_name}}) {
  join_coarse_result_count++;
  {{out_tuple_type}} {{out_tuple_name}} = {{append_func_name}}({{keyname}}, {{right_tuple_name}});
  {{inner_plan_compiled}}
});
