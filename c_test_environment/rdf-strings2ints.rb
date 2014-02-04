# Dependencies:
# raptor: http://librdf.org 
# rdf-raptor: https://github.com/ruby-rdf/rdf-raptor
require 'rdf/raptor'

$i = 0
def newid()
    r = $i
    $i+=1
    return r
end

strings = {}
RDF::Reader.open("sp2b.100t") do |reader|
    reader.each_statement do |statement|
        spo = []
        statement.to_a.each do |atr|   # can also pull out attributes with subject/predicate/object
            intid = strings[atr]    # just hash the RDF::URI object directly (or try to_s())
            if not intid then
                intid = newid()
                strings[atr] = intid
            end
            spo+=[intid]
        end

        print spo, "\n"
    end
end
        



        
