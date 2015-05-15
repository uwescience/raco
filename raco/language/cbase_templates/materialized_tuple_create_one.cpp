static {{result_type}} {{convert_func_name}}(const {{type1}}& t1) {
    {{result_type}} t;
    {% for i in range(type1numfields) %}
        t.f{{i}} = t1.f{{i}};
    {% endfor %}

    return t;
}
