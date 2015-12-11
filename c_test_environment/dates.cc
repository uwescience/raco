#include "dates.h"
#include <ctime>

namespace dates {
  const uint32_t date_format_len = 11;

  uint64_t year(std::string date) {
    return std::stoi(date.substr(0, 4));
  }
  
  uint64_t month(std::string date) {
    return std::stoi(date.substr(5, 2));
  }

  uint64_t day(std::string date) {
    return std::stoi(date.substr(8, 2));
  }

  namespace impl {
    std::string mkstrdate(const tm* timeptr) {
      char r[date_format_len];
      strftime(r, date_format_len, "%Y-%m-%d", timeptr);
      return std::string(r);
    }
  }

  std::string add(std::string t, int64_t days) {
    tm tc_ = tm(); // initialize fields to 0
    tc_.tm_year = year(t)-1900;   // years since 1900
    tc_.tm_mon = month(t)-1;     // months since january
    tc_.tm_mday = day(t);       // day of the month

    tc_.tm_mday += days;
    
    // fix up the struct tm
    std::mktime(&tc_);

    return impl::mkstrdate(&tc_);
  }
}

    
