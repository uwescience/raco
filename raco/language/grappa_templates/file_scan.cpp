{
    if (FLAGS_bin) {
        {{resultsym}} = readTuplesUnordered<{{result_type}}>( FLAGS_input_file_{{name}} + ".bin" );
    } else {
        {{resultsym}}.data = readTuples<{{result_type}}>( FLAGS_input_file_{{name}}, FLAGS_nt);
        {{resultsym}}.numtuples = FLAGS_nt;
        auto l_{{resultsym}} = {{resultsym}};
        on_all_cores([=]{ {{resultsym}} = l_{{resultsym}}; });
    }
}
