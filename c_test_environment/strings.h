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



// for array based strings
#include <array>
#include <cassert>
#define MAX_STR_LEN 25

#include <iostream>
template<size_t N, class Iterable>
std::array<char, N> to_array(const Iterable& x) {
  assert(x.size() <= N-1);
  std::array<char, N> d;
  std::copy(x.begin(), x.end(), d.data());
  *(d.data()+x.size()) = '\0'; // copy null terminator
  return d;
}

template <size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<char, N>& arr) {
  // copy to a string so null terminator is used  
  std::string s(arr.data());
  o << s;
  return o;
}

template <size_t N>
bool operator==(const std::array<char, N>& arr, const std::string& str) {
  return std::string(arr.data()).compare(str) == 0;
}
