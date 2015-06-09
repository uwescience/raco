{% block precode %}{% endblock %}
auto start_{{ident}} = walltime();
{{inner_code}}
auto end_{{ident}} = walltime();
{% block postcode %}{% endblock %}
auto runtime_{{ident}} = end_{{ident}} - start_{{ident}};
{% block printcode %}{% endblock %}
