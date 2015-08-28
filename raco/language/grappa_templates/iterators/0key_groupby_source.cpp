class {{class_symbol}} : public ZeroKeyAggregateSource<{{produce_type}}, {{state_type}}, {{combine_func}}> {
    using ZeroKeyAggregateSource<{{produce_type}}, {{state_type}}, {{combine_func}}>::ZeroKeyAggregateSource;
    protected:
        void mktuple({{produce_type}}& {{output_tuple}}, {{state_type}}& {{output_typle}}_tmp) {
            {{assignment_code}}
        }

};