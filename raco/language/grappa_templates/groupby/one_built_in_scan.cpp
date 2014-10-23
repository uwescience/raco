{% extends 'scan.cpp' %}

{% block initializer %}
std::tuple_cat({{ initializer_list|join(',') }})
{% endblock %}