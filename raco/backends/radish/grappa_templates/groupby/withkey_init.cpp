auto l_{{hashname}} = DHT_symmetric<{{keytype}},{{valtype}},hash_tuple::hash<{{keytype}}>>::create_DHT_symmetric( );
on_all_cores([=] {
  {{hashname}} = l_{{hashname}};
});
