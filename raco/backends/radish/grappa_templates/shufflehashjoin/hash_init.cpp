auto {{hashname}}_num_reducers = cores();
auto {{hashname}} = allocateJoinReducers<int64_t,{{left_type}},{{right_type}},{{out_tuple_type}}>({{hashname}}_num_reducers);
auto {{hashname}}_ctx = HashJoinContext<int64_t,{{left_type}},{{right_type}},{{out_tuple_type}}>({{hashname}}, {{hashname}}_num_reducers);
