{
    if (FLAGS_bin) {
        BinaryRelationFileReader<{{result_type}}> reader;
        {{resultsym}} = reader.read( FLAGS_input_file_{{name}} + ".bin" );
    } else if (FLAGS_jsonsplits) {
        SplitsRelationFileReader<JSONRowParser<{{result_type}},&schema_{{resultsym}}>, {{result_type}}> reader;
        {{resultsym}} = reader.read( FLAGS_input_file_{{name}} );
    } else {
        {{resultsym}}.data = readTuples<{{result_type}}>( FLAGS_input_file_{{name}}, FLAGS_nt);
        {{resultsym}}.numtuples = FLAGS_nt;
        auto l_{{resultsym}} = {{resultsym}};
        on_all_cores([=]{ {{resultsym}} = l_{{resultsym}}; });
    }
}
