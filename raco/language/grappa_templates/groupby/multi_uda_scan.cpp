{% extends 'scan.cpp' %}

{# depends on materialized_tuple_ref constructor of std::tuple #}
{% block initializer %}{{ super() }}, {{mapping_var_name}}.second.to_tuple(){% endblock %}
