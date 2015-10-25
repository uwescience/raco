{% extends "group_timing.cpp" %}
{% block printcode %}
{{timer_metric}} += runtime_{{ident}};
VLOG(1) << "pipeline group {{ident}}: " << runtime_{{ident}} << " s";
{% endblock %}

{% block precode %}Grappa::Metrics::reset();
{{tracing_on}}{% endblock %}

{% block postcode %}{{tracing_off}}{% endblock %}
