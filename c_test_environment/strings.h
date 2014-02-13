#pragma once

#include <string>
#include <map>
#include <vector>

class StringIndex {
  private:
    std::vector<std::string> strings;
    std::vector<int64_t> indices;

  public: 
    size_t size() const;
    StringIndex(std::map<std::string, int64_t> mapping);
    StringIndex();
    int64_t string_lookup(std::string s);
};


StringIndex build_string_index(std::string indexfn);

namespace QueryUtils {

  template <typename Iter, typename T>
    Iter binary_search(Iter begin, Iter end, const T& key) {
      Iter i = std::lower_bound(begin, end, key);

      
      if (i != end && (key == *i)) {
        return i; 
      } else {
        return end; 
      }
  }
}

