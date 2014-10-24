{% extends 'scan.cpp' %}

{% block initializer %}{{ super() }}, std::make_tuple({{mapping_var_name}}.second){% endblock %}
