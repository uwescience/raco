#pragma once

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

struct relationInfo *inhale(char *path, struct relationInfo *relInfo);
struct relationInfo *binary_inhale(char *path, struct relationInfo *relInfo);

void printrelation(struct relationInfo *R);

#define ZAPPA

