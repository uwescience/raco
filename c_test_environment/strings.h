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
    StringIndex(const std::map<std::string, int64_t>& mapping);
    StringIndex();
    int64_t string_lookup(const std::string& s) const;
};


StringIndex build_string_index(const std::string& indexfn);

namespace QueryUtils {

  template <typename Iter, typename T>
    Iter binary_search(Iter begin, Iter end, const T& key) {
      auto i = std::lower_bound(begin, end, key);
      
      if (i != end && (key == *i)) {
        return i; 
      } else {
        return end; 
      }
  }
}

