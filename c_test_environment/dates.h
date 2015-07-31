#include <cstdint>
#include <string>
#include <array>

// date format is a string YYYY-MM-DD
//                         0123456789
// "1998-01-01"

namespace dates {

  uint64_t year(std::string date); 
  uint64_t month(std::string date);
  uint64_t day(std::string date);

// for passing an array
// TODO implicit conversion to avoid this code, https://github.com/uwescience/raco/issues/454
  template <size_t N>
  uint64_t year(std::array<char, N> date) {
    year(std::string(date.data()));
  }
  template <size_t N>
  uint64_t month(std::array<char, N> date) {
    month(std::string(date.data()));
  }
  template <size_t N>
  uint64_t day(std::array<char, N> date) {
    day(std::string(date.data()));
  }

  std::string add(std::string t, int64_t days);
}

    
