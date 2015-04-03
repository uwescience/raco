for (auto {{right_tuple_name}} : lookup({{hashname}}, {{keyval}})) {
    auto {{out_tuple_name}} = create({{keyname}}, {{right_tuple_name}});
    {{inner_plan_compiled}}
}
