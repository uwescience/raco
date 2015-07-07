#include <cstdint>
#include <string>

// date format is a string YYYY-MM-DD
//                         0123456789
// "1998-01-01"

namespace dates {

  uint64_t year(std::string date); 
  uint64_t month(std::string date);
  uint64_t day(std::string date);

  std::string add(std::string t, int64_t days);
}

    
