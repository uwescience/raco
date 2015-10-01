class {{class_symbol}} : public ZeroKeyAggregateSource<{{produce_type}}, {{state_type}}, &{{combine_func}}> {
    using ZeroKeyAggregateSource<{{produce_type}}, {{state_type}}, {{combine_func}}>::ZeroKeyAggregateSource;
    protected:
        void mktuple({{produce_type}}& {{produce_tuple_name}}, {{state_type}}& {{state_tuple_name}}) {
            {{assignment_code}}
        }

};