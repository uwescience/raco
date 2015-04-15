for (auto it={{hashname}}.begin(); it!={{hashname}}.end(); it++) {
    {{output_tuple_type}} {{output_tuple_name}}(it->first, it->second);
    {{inner_code}}
}
