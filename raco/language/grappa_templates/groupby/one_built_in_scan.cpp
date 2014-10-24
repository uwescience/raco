{% extends 'scan.cpp' %}

{% block initializer %}std::tuple_cat({{ super() }}, {{mapping_var_name}}.second){% endblock %}