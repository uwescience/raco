{% extends 'scan.cpp' %}

{# depends on materialized_tuple_ref constructor of std::tuple #}
{% block initializer %}std::tuple_cat({{ super() }}, {{mapping_var_name}}.second.to_tuple()){% endblock %}
