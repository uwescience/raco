class {{class_symbol}} : public HashJoinSink{{side}}<{{keytype}},
                                                     {{left_tuple_type}},
                                                     {{right_tuple_type}},
                                                     hash_tuple::hash<{{keytype}}>, &{{pipeline_sync}}> {
    using HashJoinSink{{side}}<{{keytype}},
                                                     {{left_tuple_type}},
                                                     {{right_tuple_type}},
                                                     hash_tuple::hash<{{keytype}}>, &{{pipeline_sync}}>::HashJoinSink{{side}};
    protected:
      {{keytype}} mktuple({% if side == 'Right' %}
                                      {{right_tuple_type}}
                                      {% else %}
                                      {{left_tuple_type}}
                                      {% endif %}
                                      &{{input_tuple_name}}) {

            return {{keyval}};
      }
};

