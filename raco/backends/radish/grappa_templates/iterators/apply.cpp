class {{class_symbol}} : public Apply<{{consume_type}}, {{produce_type}}> {
    using Apply<{{consume_type}}, {{produce_type}}>::Apply;
    protected:
      void apply({{produce_type}}& {{produce_tuple_name}}, {{consume_type}}& {{consume_tuple_name}}) {
        {{statements}}
      }
};