auto start_{{ident}} = walltime();
{% block printstart %} << "timestamp {{ident}} start " << std::setprecision(15) << start_{{ident}}{% endblock %}

{{inner_code}}
auto end_{{ident}} = walltime();
auto runtime_{{ident}} = end_{{ident}} - start_{{ident}};
{% block printruntime %} << "pipeline {{ident}}: " << runtime_{{ident}} << " s"{% endblock %}

{% block printend %} << "timestamp {{ident}} end " << std::setprecision(15) << end_{{ident}}{% endblock %}

