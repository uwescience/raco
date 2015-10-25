{{state_type}} {{name}}_update(const {{state_type}}& state, const {{input_type}}& {{input_tuple_name}}) {
    {% for u in update_updates %}
    {{ u }}
    {% endfor %}
    return {{state_type}}(std::make_tuple({{ update_state_vars|join(',') }}));
}