auto l_{{sid}} = string_index.string_lookup({{st}});
on_all_cores([=] { {{sid}} = l_{{sid}}; });
