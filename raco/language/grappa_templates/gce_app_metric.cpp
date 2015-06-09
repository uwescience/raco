{% extends "define_metric.cpp" %}

{% block type %}CallbackMetric<int64_t>{% endblock %}

{% block name %}app_{{pipeline_id}}_gce_incomplete{% endblock %}

{% block init %}[] {
  return {{global_syncname}}.incomplete();
}{% endblock %}

