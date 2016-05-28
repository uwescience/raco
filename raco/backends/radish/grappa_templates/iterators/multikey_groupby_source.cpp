class {{class_symbol}} : public AggregateSource<{{produce_type}}, {{keytype}}, {{state_type}}, {{input_type}}> {
    using AggregateSource<{{produce_type}}, {{keytype}}, {{state_type}}, {{input_type}}>::AggregateSource;

    private:
        typedef AggregateSource<{{produce_type}}, {{keytype}}, {{state_type}}, {{input_type}}>::map_output_t map_output_t;

    protected:
        void mktuple({{produce_type}}& {{produce_tuple_name}}, map_output_t& {{mapping_var_name}}) {
            {{produce_type}} {{produce_tuple_name}}_tmp(std::tuple_cat({{mapping_var_name}}.first, {{mapping_var_name}}.second.to_tuple()));
            {{produce_tuple_name}} = {{produce_tuple_name}}_tmp;
        }
};