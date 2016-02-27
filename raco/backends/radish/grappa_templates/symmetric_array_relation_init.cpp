{
    decltype({{sym}}) l_{{sym}};
    l_{{sym}}.data = Grappa::symmetric_global_alloc<aligned_vector<{{tuple_type}}>>();
    l_{{sym}}.numtuples = 0; // starts empty, but it could get filled
                           // Often we may not want to bother counting, so this
                           // field may become incoherent.

    // make it available everywhere
    on_all_cores([=] {
        {{sym}} = l_{{sym}};
    });
}