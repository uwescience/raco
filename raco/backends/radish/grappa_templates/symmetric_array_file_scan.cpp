if (FLAGS_bin) {
BinaryRelationFileReader<{{result_type}},
                           aligned_vector<{{result_type}}>,
                           SymmetricArrayRepresentation<{{result_type}}>> reader;
                           // just always broadcast the name to all cores
                           // although for some queries it is unnecessary
                           auto l_{{resultsym}} = reader.read( FLAGS_input_file_{{name}} + ".bin" );
                           on_all_cores([=] {
                                {{resultsym}} = l_{{resultsym}};
                           });

                           } else {

                           CHECK(false) << "only --bin=true supported for symmetric array repr";

                           }
