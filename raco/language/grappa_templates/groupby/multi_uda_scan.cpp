{% extends 'scan.cpp' %}

{% block initializer %}
{{ initializer_list|join(',') }}
{% endblock %}
