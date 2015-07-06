DEFINE_string(input_file_{{name}}, "{{name}}", "Input file");
{% block input_relation %}
{% endblock %}
std::vector<std::string> schema_{{resultsym}} = { {% for c in colnames %}"{{c}}",{% endfor %} };
