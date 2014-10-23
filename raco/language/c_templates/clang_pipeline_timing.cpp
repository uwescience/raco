{% extends "pipeline_timing.cpp" %}

{% block printstart %}
std::cout {{ super() }} << std::endl;
{% endblock %}

{% block printruntime %}
std::cout {{ super() }} << std::endl;
{% endblock %}

{% block printend %}
std::cout {{ super() }} << std::endl;
{% endblock %}
