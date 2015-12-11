#include "dates.h"
#include <iostream>
#include <string>

int main() {
  // test string manipulations from TPC-H Q1
  std::string d = "1998-12-01";
  std::string ct = dates::add(d, -60);
  
  std::cout << d << " " << ct << std::endl;

  std::string in = "1998-01-01";
  std::cout << in << " <= " << ct << " | " << (in <= ct) << std::endl;

  std::string in2 = "1998-11-29";
  std::cout << in2 << " <= " << ct << " | " << (in2 <= ct) << std::endl;
}

