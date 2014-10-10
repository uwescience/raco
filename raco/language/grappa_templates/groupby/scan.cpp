{{hashname}}->forall_entries<&{{pipeline_sync}}>([=](std::pair<const {{keytype}},{{emit_type}}>&{{mapping_var_name}}) {
    {{output_tuple_type}} {{output_tuple_name}}({% block initializer %}{% endblock %});
    {{inner_code}}
});

