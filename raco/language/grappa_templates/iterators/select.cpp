class {{class_symbol}} : public Select<{{consume_type}}, {{produce_type}}> {
    using Select<{{produce_type}}, {{produce_type}}>::Select;
    protected:
        bool predicate({{produce_type}}& {{consume_tuple_name}}) {
            return {{expression}};
        }
};