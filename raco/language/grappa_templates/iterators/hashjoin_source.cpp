class {{class_symbol}} : public HashJoinSource<{{keytype}},
                                                {{left_tuple_type}},
                                                {{right_tuple_type}},
                                                hash_tuple::hash<{{keytype}}>,
{{out_tuple_type}}> {

    using HashJoinSource<{{keytype}},
                                                {{left_tuple_type}},
                                                {{right_tuple_type}},
                                                hash_tuple::hash<{{keytype}}>, {{out_tuple_type}}>::HashJoinSource;

    protected:
        {{out_tuple_type}} mktuple({{left_tuple_type}}& {{left_name}}, {{right_tuple_type}}& {{right_name}}) {
            return {{append_func_name}}({{left_name}}, {{right_name}});
        }
};