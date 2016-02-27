// recycle result vector
on_all_cores([=] {
    swap_refs({{symsrc}}.data, {{symsink}}.data);
    {{symsink}}.data->vector.clear();
});
