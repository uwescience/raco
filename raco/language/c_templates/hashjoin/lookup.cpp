for (auto {{right_tuple_name}} : lookup({{hashname}}, {{keyval}})) {
    auto {{out_tuple_name}} = {{append_func_name}}({{keyname}}, {{right_tuple_name}});
    {{inner_plan_compiled}}
}
