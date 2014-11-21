MapReduce::forall_symmetric<&{{pipeline_sync}}>({{hashname}}, &JoinReducer<int64_t,{{left_type}},{{right_type}},{{out_tuple_type}}>::resultAccessor, [=]({{out_tuple_type}}& {{out_tuple_name}}) {
{{inner_code_compiled}}
});
