{{state_type}} {{name}}_init() {
    {% for u in init_updates %}
    {{u}}
    {% endfor %}

    return {{state_type}}( std::make_tuple({{ init_state_vars|join(',') }}) );
}