auto {{output_tuple_name}}_tmp = reduce<{% block templateargs %}{% endblock %}>({{hashname}});

{% block output %}{% endblock %}

{{inner_code}}
