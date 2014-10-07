auto start_{{ident}} = walltime();
{{grpcode}}
auto end_{{ident}} = walltime();
auto runtime_{{ident}} = end_{{ident}} - start_{{ident}};
std::cout << "pipeline group {{ident}}: "
          << runtime_{{ident}}
          << " s" << std::endl;

