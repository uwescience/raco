{% extends 'scan.cpp' %}

{# depends on materialized_tuple_ref constructor of std::tuple #}
{% block initializer %}std::tuple_cat({{ super() }}, std::make_tuple({{mapping_var_name}}.second)){% endblock %}