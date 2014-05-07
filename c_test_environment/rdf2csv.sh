# steps:
# create ^A separated triples file
# ./rdf2csv rdffile
#
# index the triples file (*.i index version, *.index index of strings)
# ruby triples-strings2ints.rb rdffile.str

rdffile=$1
prefixlines=9
linesperpart=14000009 
./filesplit $rdffile $linesperpart $prefixlines
./build_rdf_from_parts.sh $rdffile
