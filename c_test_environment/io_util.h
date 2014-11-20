#pragma once
#include <string>
#include <sstream>
#include <sstream>
#include <vector>
#include <fstream>

// How to use the I/O utilities:
// 1) Inhale a particular file. Right now, expected to be a space separated
//    char *filePath = "/scratch/tmp/...";
//    struct relationInfo relInfo;
//    struct relationInfo *ptr = binary_inhale(filePath, &relInfo);
//    OR
//    struct relationInfo *ptr = inhale(filePath, &relInfo);
// 2) Manipulate the relation as you see fit.
//    ...
// 3) Free the memory for the relation
//    free(relInfo.data);

double timer();
  
class RangeIter;
class RangeIter {
  private:
    uint64_t num;
    uint64_t next;
  public:
    RangeIter(uint64_t num, bool asEnd=false);
    
    uint64_t operator*();

    RangeIter& operator++();

    bool notequal(const RangeIter& o) const;
};

class RangeIterable {
  private:
    uint64_t num;
  public:
    RangeIterable(uint64_t num);

    RangeIter begin();
    RangeIter end();
};

  
struct relationInfo {
  uint64 tuples;
  uint64 fields;
  int64 *relation;

  RangeIterable range() {
    return RangeIterable(tuples);
  }
};
      
bool operator!=(const RangeIter& o1, const RangeIter& o2);
bool operator==(const RangeIter& o1, const RangeIter& o2);

struct relationInfo *inhale(const char *path, struct relationInfo *relInfo);
struct relationInfo *binary_inhale(const char *path, struct relationInfo *relInfo);

void printrelation(struct relationInfo *R);


template<typename T>
void tuplesFromAscii(const char *path, std::vector<T>& buf) {
  std::string pathst(path);
  std::ifstream testfile(pathst, std::ifstream::in);

  std::string line;
  while (std::getline(testfile,line)) {
    std::istringstream ss(line);
    buf.push_back(T::fromIStream(ss)); 
  }
}
    

#define ZAPPA

