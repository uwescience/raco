static {{result_type}} {{append_func_name}}(const {{type1}}& t1, const {{type2}}& t2) {
    {{result_type}} t;
    {% for i in range(type1numfields) %}
        t.f{{i}} = t1.f{{i}};
    {% endfor %}

    {% for i in range(type2numfields) %}
        t.f{{i+type1numfields}} = t2.f{{i}};
    {% endfor %}

    return t;
}
