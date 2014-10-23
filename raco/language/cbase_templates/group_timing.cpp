auto start_{{ident}} = walltime();
{{inner_code}}
auto end_{{ident}} = walltime();
auto runtime_{{ident}} = end_{{ident}} - start_{{ident}};
{% block printcode %}{% endblock %}
