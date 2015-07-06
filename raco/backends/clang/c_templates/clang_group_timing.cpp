{% extends "group_timing.cpp" %}
{% block printcode %}
std::cout << "pipeline group {{ident}}: "
          << runtime_{{ident}}
          << " s" << std::endl;
{% endblock %}
