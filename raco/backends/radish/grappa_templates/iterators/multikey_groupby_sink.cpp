class {{class_symbol}} : public AggregateSink<{{consume_type}}, {{keytype}}, {{state_type}}, &{{pipeline_sync}}> {
using AggregateSink<{{consume_type}}, {{keytype}}, {{state_type}}, &{{pipeline_sync}}>::AggregateSink;
protected:
    {{keytype}} mktuple({{consume_type}}& {{consume_tuple_name}}) {
        return std::make_tuple({{ keygets|join(',') }});
    }

};