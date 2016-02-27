// recycle result vector
swap_refs({{symsrc}}.data, {{symsink}}.data);
on_all_cores([=] {
    {{symsink}}.data->vector.clear();
});
