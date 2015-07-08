{{comment}}
{{hashname}}->update<&{{pipeline_sync}}, {{input_type}}, &{{update_func}},&{{init_func}}>(std::make_tuple({{ keygets|join(',') }}), {{update_val}});
