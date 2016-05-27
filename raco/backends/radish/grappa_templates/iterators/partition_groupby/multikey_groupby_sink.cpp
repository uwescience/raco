class {{class_symbol}} : public AggregatePartitionSink<{{consume_type}}, {{keytype}}, {{state_type}}> {
using AggregatePartitionSink<{{consume_type}}, {{keytype}}, {{state_type}}>::AggregatePartitionSink;
protected:
    {{keytype}} mktuple({{consume_type}}& {{consume_tuple_name}}) {
        return std::make_tuple({{ keygets|join(',') }});
    }

};