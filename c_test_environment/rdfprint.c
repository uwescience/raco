#include <stdio.h>
#include <stdlib.h>
#include <raptor2.h>

/* rdfprint.c: print triples from parsing RDF/XML */
int i=0;
static void
print_triple(void* user_data, raptor_statement* triple) 
{
  //raptor_statement_print_as_ntriples(triple, stdout);
  //fputc('\n', stdout);
  i++; 
  char delim = 1;
  printf("%s%c%s%c%s\n", raptor_term_to_string(triple->subject), delim, 
                         raptor_term_to_string(triple->predicate), delim,
                         raptor_term_to_string(triple->object)); 
  //if (i%10000==0) printf("i=%d\n", i);
}

int
main(int argc, char *argv[])
{
  raptor_world *world = NULL;
  raptor_parser* rdf_parser = NULL;
  unsigned char *uri_string;
  raptor_uri *uri, *base_uri;

  world = raptor_new_world();

  rdf_parser = raptor_new_parser(world, "turtle");

  raptor_parser_set_statement_handler(rdf_parser, NULL, print_triple);

  if (argc < 2) {
    printf("usage: %s <rdf file>\n", argv[0]);
    exit(1);
  }

  //printf("uri_string = ...\n");
  uri_string = raptor_uri_filename_to_uri_string(argv[1]);
  //printf("uri = ...\n");
  uri = raptor_new_uri(world, uri_string);
  //printf("base_uri = ...\n");
  base_uri = raptor_uri_copy(uri);

  //printf("parsing ...\n");
  raptor_parser_parse_file(rdf_parser, uri, base_uri);
  //printf("done parsing ...\n");

  raptor_free_parser(rdf_parser);

  raptor_free_uri(base_uri);
  raptor_free_uri(uri);
  raptor_free_memory(uri_string);

  raptor_free_world(world);
  //printf("i=%d\n", i);

  return 0;
}
