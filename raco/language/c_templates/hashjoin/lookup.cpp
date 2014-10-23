for (auto {{right_tuple_name}} : lookup({{hashname}}, {{keyval}})) {
    auto {{out_tuple_name}} = {{out_tuple_type}}::create({{keyname}}, {{right_tuple_name}});
    {{inner_plan_compiled}}
}
