auto {{hashname}} = DHT_symmetric_generic<{{keytype}},{{valtype}},{{update_val_type}},hash_tuple::hash<{{keytype}}>>::create_DHT_symmetric(&{{update_func}}, &{{init_func}});
