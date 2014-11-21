{{hashname}}.insert_lookup_iter_{{side}}<&{{global_syncname}}>({{keyval}}, {{keyname}}, [=]({{other_tuple_type}} {{valname}}) {
  join_coarse_result_count++;
  {{out_tuple_type}} {{out_tuple_name}} = {{out_tuple_type}}::create<{{left_type}}, {{right_type}}> ({{left_name}}, {{right_name}});
  {{inner_plan_compiled}}
});
