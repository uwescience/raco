for (auto it={{hashname}}.begin(); it!={{hashname}}.end(); it++) {
    {{output_tuple_type}} {{output_tuple_name}}(it->first.first, it->first.second, it->second);
    {{inner_code}}
}
