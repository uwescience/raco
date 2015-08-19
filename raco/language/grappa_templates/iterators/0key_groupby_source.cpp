class {{class_symbol}} : public ZeroKeyAggregateSource<{{produce_type}}, {{state_type}}, {{combine_func}}> {
    using ZeroKeyAggregateSource<{{produce_type}}, {{state_type}}, {{combine_func}}>::ZeroKeyAggregateSource;
    protected:
        void mktuple({{produce_type}}& dest, {{state_type}}& src) {
            {{assignment_code}}
        }

};