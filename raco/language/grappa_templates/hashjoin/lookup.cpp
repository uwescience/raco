{{hashname}}.lookup_iter<&{{pipeline_sync}}>( {{keyval}}, [=]({{right_tuple_type}}& {{right_tuple_name}}) {
  join_coarse_result_count++;
  {{out_tuple_type}} {{out_tuple_name}} = {{out_tuple_type}}::create<{{input_tuple_type}}, {{right_tuple_type}}>({{keyname}}, {{right_tuple_name}});
  {{inner_plan_compiled}}
});
