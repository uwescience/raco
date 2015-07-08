if (FLAGS_bin) {
BinaryRelationFileReader<{{result_type}},
                           aligned_vector<{{result_type}}>,
                           SymmetricArrayRepresentation<{{result_type}}>> reader;
                           {{resultsym}} = reader.read( FLAGS_input_file_{{name}} + ".bin" );
                           } else {

                           CHECK(false) << "only --bin=true supported for symmetric array repr";

                           }
