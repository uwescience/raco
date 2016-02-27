// recycle result vector
swap({{symsrc}}.data, {{symsink}}.data);
on_all_cores([=] {
    {{symsink}}.data->vector.clear();
});
