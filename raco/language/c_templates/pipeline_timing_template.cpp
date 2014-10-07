auto start_{{ident}} = walltime();
std::cout << "timestamp {{ident}} start "
          << std::setprecision(15)
          << start_{{ident}} << std::endl;
{{inner_code}}
auto end_{{ident}} = walltime();
auto runtime_{{ident}} = end_{{ident}} - start_{{ident}};
std::cout << "pipeline {{ident}}: "
          << runtime_{{ident}} << " s"
          << std::endl;
std::cout << "timestamp {{ident}} end "
          << std::setprecision(15)
          << end_{{ident}} << std::endl;

