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



static const int64_t BIN_SEARCH_NOT_FOUND  = -1;

template <typename StringArray>
int64_t binary_search(StringArray arr, std::string key, int64_t imin, int64_t imax) {
  while (imax >= imin) {
    int64_t imid = (imin + imax) / 2;

    int64_t cmp = key.compare(arr[imid]);

    if(cmp == 0) {
      // key found
      return imid; 
    } else if (cmp > 0) {
      // key larger, so search upper partition
      imin = imid + 1;
    } else {       
      // key smaller so search lower partition
      imax = imid - 1;
    }
  }
  // key not found
  return BIN_SEARCH_NOT_FOUND;
}
