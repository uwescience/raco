# Dependencies:
# raptor: http://librdf.org 
# rdf-raptor: https://github.com/ruby-rdf/rdf-raptor
require 'rdf/raptor'

"""
TODO!!! Retest with below
Prefixes are now fixed in 1.2.0 of rdf/raptor!
https://github.com/ruby-rdf/rdf-raptor/issues/13#issuecomment-34163991
"""

$inputfile = ARGV[0]

$extras = false
if ARGV.length > 1 and ARGV[1]=='extras' then
  $extras = true
end
$int_index = true
if ARGV.length > 1 and ARGV[1]=='str' then
    $int_index = false
end

"""
Some junk trying to output prefix reduced versions of strings,
but need to figure out how to define a new vocabulary with its own uri.

For some reason the Reader keeps its prefixes internal.
If you could get them, you would pass to a Writer as writer.options[:prefixes] = prefs,
then the writer will use those
"""
$prefixes = [RDF::DC11,
            RDF::DC,
            RDF::FOAF,
            RDF::XSD]
$memoized = {}
def uriToPrefixedStr(uri)
   r = $memoized[uri]
   if not r then
      r = ""
      $prefixes.each do |pref|
          pref_pat = Regexp.new (pref.to_uri.to_s)   # or path only
          prefix_match = pref_pat.match(uri.to_s) 
          if prefix_match then
              r = "#{pref.__prefix__.to_s}:#{prefix_match.post_match}"
              $memoized[uri] = r
              break
          end
      end
   end

   # give up if no prefix match found
   if r.empty? then
       return uri.to_s
   end
   
   return r 
end




$i = 0
def newid()
    r = $i
    $i+=1
    return r
end

if $int_index then
    strings = {}
    # including the prefixes is silly and unnecessary
    open("#{$inputfile}.i", 'w') do |writer|
        RDF::Reader.open($inputfile) do |reader|
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

                writer.write("#{spo.join(" ")}\n")
            end
        end

    end

    # encoding of map is is (string=>index) -> (implicitly linenumber) 
    open("#{$inputfile}.index", 'w') do |writer|
        strings.each_pair do |k,v|
            writer.write("#{k.to_s}\n")
        end
    end  
else 
    open("#{$inputfile}.str", 'w') do |writer|
        RDF::Reader.open($inputfile) do |reader|
            reader.each_statement do |statement|
                spo = []
                statement.to_a.each do |atr|   # can also pull out attributes with subject/predicate/object
                    spo+=[atr.to_s]
                end

                writer.write("#{spo.join(1.chr)}\n")
            end
        end

    end
end
    

if $extras then

  # encoding of map is is (string=>index) -> (string linenumber) 
  open("#{$inputfile}.index_human", 'w') do |writer|
    strings.each_pair do |k,v|
      writer.write("#{k.to_s} #{v}\n")
    end
  end  

  # output a version of the mapping with prefix-shortened rdf strings
  open("#{$inputfile}.index_slim", 'w') do |writer|
    strings.each_pair do |k,v|
      pk = uriToPrefixedStr(k)
      writer.write("#{pk.to_s} #{v}\n")
    end
  end

end



"""
def somethingThatWorks():
   element = a term from the triple (a URI)
   vocab = RDF::FOAF

    element.starts_with? vocab   => true means prefix or equal 
    """

    """alternatively so I can hash
    element.parent == vocab   (or vocab.to_uri)
    But this is less robuts, for e.g.
    RDF::RDFS has hashtags which break parent
    """
