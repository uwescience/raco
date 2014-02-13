#include "strings.h"

#include <stdexcept>
#include <fstream>

  

size_t StringIndex::size() const {
  return strings.size();
}

StringIndex::StringIndex(const std::map<std::string, int64_t>& mapping) : strings(), indices() {
  // mapping stores the strings in sorted order
  // so this forms sorted vectors
  for (auto p : mapping) {
    strings.push_back(p.first);
    indices.push_back(p.second);
  }
}

// This integer represents strings not in the database
const int64_t DB_NON_EXISTANT_STRING = -1;
int64_t StringIndex::string_lookup(const std::string& s) const {
  // TODO: use trie structure instead of binary search
  
  auto ifound = QueryUtils::binary_search( this->strings.begin(), this->strings.end(), s);
  if (ifound == this->strings.end()) {
    return DB_NON_EXISTANT_STRING;
  } else {
    auto ind = (ifound - this->strings.begin());
    return this->indices[ind];
  }
}

StringIndex::StringIndex() : strings(), indices() {}

StringIndex build_string_index(const std::string& indexfn) {
  std::map<std::string, int64_t> str2int;
  std::ifstream file( indexfn );
  std::string line;
  int64_t ln = 0;
  while (getline( file, line )) {
    str2int[line] = ln++;
  }

  return StringIndex(str2int);
}
  
