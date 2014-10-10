{% extends "pipeline_timing.cpp" %}

{% block printstart %}VLOG(1) {{ super() }};{% endblock %}

{% block printruntime %}VLOG(1) {{ super() }};{% endblock %}

{% block printend %}VLOG(1) {{ super() }};{% endblock %}