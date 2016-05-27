{{comment}}
auto {{output_tuple_name}}_tmp = reduce<{% block templateargs %}{% endblock %}>({{hashname}});

{% block output %}{% endblock %}

{{inner_code}}

// putting a wait here satisfies the invariant that inner code depends
// on global synchronization by the pipeline source
{{pipeline_sync}}.wait();

