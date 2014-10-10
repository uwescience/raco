{
    tuple_graph tg;
    tg = readTuples( "{{name}}" );

    FullEmpty<GlobalAddress<Graph<Vertex>>> f1;
    privateTask( [&f1,tg] {
      f1.writeXF( Graph<Vertex>::create(tg, /*directed=*/true) );
    });
    auto l_{{resultsym}}_index = f1.readFE();

    on_all_cores([=] {
      {{resultsym}}_index = l_{{resultsym}}_index;
    });
}

