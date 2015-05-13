{% extends '0key_output.cpp' %}

{% block templateargs %}{{state_type}}, counter, &{{update_func}}, &get_count{% endblock %}

{% block output %}
{{output_tuple_type}} {{output_tuple_name}};
{{output_tuple_set_func}} = {{output_tuple_name}}_tmp;
{% endblock %}

