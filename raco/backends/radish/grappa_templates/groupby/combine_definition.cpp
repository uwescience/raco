{{state_type}} {{name}}_combine(const {{state_type}}& state0, const {{state_type}}& state1) {
    {% for c in combine_updates %}
    {{ c }}
    {% endfor %}
    return {{state_type}}(std::make_tuple({{ combine_state_vars|join(',') }}));
}
