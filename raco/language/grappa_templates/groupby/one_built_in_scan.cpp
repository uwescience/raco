{% extends 'scan.cpp' %}

{% block initializer %}std::tuple_cat({{ super() }}, make_tuple({{mapping_var_name}}.second)){% endblock %}