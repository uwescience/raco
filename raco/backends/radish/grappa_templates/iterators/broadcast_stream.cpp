class {{class_symbol}} : public BroadcastTupleStream<{{left_type}}, {{right_type}}, {{output_type}}> {
    using BroadcastTupleStream<{{left_type}}, {{right_type}}, {{output_type}}>::BroadcastTupleStream;
    protected:
        void mktuple({{output_type}}& {{output_name}}, {{left_type}}& l, {{right_type}}& r) {
              {{output_name}} = {{append_func_name}}(l, r);
        }
};