{% extends '0key_output.cpp' %}

{% block templateargs %}{{state_type}}, counter<{{state_type}}>, &{{combine_func}}, &get_count<{{state_type}}>{% endblock %}

{% block output %}
{{output_tuple_type}} {{output_tuple_name}};
{{output_tuple_set_func}} = {{output_tuple_name}}_tmp;
{% endblock %}

