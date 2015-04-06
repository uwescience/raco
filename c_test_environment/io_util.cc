#include <stdio.h>
#include <stdlib.h>     // for exit()
#include <fcntl.h>      // for open()
#include <unistd.h>     // for close()
#include <sys/stat.h>   // for fstat()
#include <ctype.h>      // for isdigit()
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <sys/file.h>

#ifdef __MTA__
#include <machine/runtime.h>
#include <luc/luc_common.h>
#include <snapshot/client.h>
#include <sys/mta_task.h>


typedef int int64;
typedef unsigned uint64;
#else
#include <sys/time.h>

typedef long int int64;
typedef long unsigned uint64;
#endif

#include "io_util.h"

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


double timer() {
#ifdef __MTA__
  return double(mta_get_clock(0))/mta_clock_freq();
#else
  struct timeval time;
  gettimeofday(&time, 0);
  return (time.tv_sec*1000000 + time.tv_usec)/1000000.0;
#endif
}


#define ASSERT(expression, message)                    \
{ if (!(expression)) {                                 \
  fprintf(stderr,"Assertion failed: %s\n", (message)); \
  exit(-1); }}


#pragma mta inline
uint64 convert(char *p) {
  char c = *p++;
  bool negate = false;
  if (c == '-') {
    negate = true;
    c = *p++;
  }
  //uint64 nn = atoi(p);
  //printf("|%ld|", nn);
  uint64 n = 0;
  //printf("before conversion: ");
  //char *start = p-2;
  while (isdigit(c)) {
    //printf("%c", c);
    n = n*10 + c - '0';
    c = *p++;
  }
  //uint64 test = 394239671352001345LL;
  //if (test==n) { printf(" equal\n"); }
  //printf("\n");
  /*
  uint64 pow=1;
  for (p=p-2;p!=start;p--) {
    c = *p;
    printf("%c\t", c);
    printf("%d\n", (c - '0')*pow);
    printf("%d\n", pow);
    n = n + (c - '0')*pow;
    pow *= 10;
  }
  printf("\n");
  */
  //printf("\t%lld", test);
  // printf("\n");
  return n;
//  return negate ? -n : n;
}


uint64 tuples;
uint64 fields;

int64 *relation;


struct relationInfo * inhale(const char *path, struct relationInfo *relInfo) {
  printf("\tinhaling %s\n", path);
#ifdef ZAPPA
  double start = timer();
#ifdef __MTA__
  snap_stat_buf stats;
  ASSERT(snap_stat(path, SNAP_ANY_SW, &stats, 0) == SNAP_ERR_OK, "snap_stat failed");
  uint64 bytes = stats.st_size;
  char *buf = (char *) malloc(bytes);
  ASSERT(buf, "failed to allocate memory for buf");
  ASSERT(snap_restore(path, buf, bytes, 0) == SNAP_ERR_OK, "snap_read failed");
#else
  int f = open(path, O_RDONLY);
  ASSERT(f >= 0, strerror(errno));
  struct stat stats;
  ASSERT(fstat(f, &stats) >= 0, strerror(errno));
  uint64 bytes = stats.st_size;
  printf("\tfile is %lu bytes\n", bytes);
  char *buf = (char *) malloc(bytes);
  ASSERT(buf, "failed to allocate memory for buf");
  ASSERT(read(f, buf, bytes) >= 0, strerror(errno));
  ASSERT(close(f) >= 0, strerror(errno));
#endif
  double finish = timer();
  printf("\t%f seconds for all I/O\n", finish - start);
  printf("\tFirst tuple: ");

  start = timer();
  fields = 1;
  for (uint64 i = 0; buf[i] != '\n'; i++) {
    if (buf[i] == ' ') {
      printf(", ");
      fields++;
    } else {
      printf("%c", buf[i]);
    }
  }

  tuples = 0;
  for (uint64 i = 0; i < bytes; i++) {
    char c = buf[i];
    if (c == '\n') {
      buf[i] = ' ';
      tuples++;
    }
  }

  printf("\n\tfound %lu tuples and %lu fields\n", tuples, fields);
  printf("\ttrying to allocate %f Mbytes\n", 1.0*tuples*fields*sizeof(uint64)/(1<<20));
  relation = (int64 *) malloc(tuples*fields*sizeof(uint64));
  ASSERT(relation, "failed to allocate memory for relation");

  uint64 numbers = 0;
  if (buf[0] != ' ')
    relation[numbers++] = 0;
#pragma mta assert nodep
  for (int i = 1; i < bytes; i++)
    if (buf[i - 1] == ' ' && buf[i] != ' ')
      relation[numbers++] = i;

  printf("\tfound %ld numbers\n", numbers);
  ASSERT(numbers == fields*tuples, "numbers should equal fields*tuples");

#pragma mta assert nodep
  for (uint64 i = 0; i < numbers; i++) {
    relation[i] = convert(buf + relation[i]);
    //printf("converted: %lu\n", relation[i]);
  }

  free(buf);
  finish = timer();
  printf("\t%f seconds to convert\n", finish - start);
  printf("\t%lu fields\n", fields);
  printf("\t%lu tuples\n", tuples);

  relInfo->tuples = tuples;
  relInfo->fields = fields;
  relInfo->relation = relation;
#endif // ZAPPA
  return relInfo;
}



#pragma mta trace level 0
#pragma mta no inline
struct relationInfo * binary_inhale(const char *path, struct relationInfo *relInfo) {
  double start = timer();
#ifdef __MTA__
  snap_stat_buf stats;
  ASSERT(snap_stat(path, SNAP_ANY_SW, &stats, 0) == SNAP_ERR_OK, "snap_stat failed");
  off_t bytes = (stats.st_size + 24); // *3/4;
  relation = (int64 *) malloc(bytes);
  ASSERT(relation, "failed to allocate memory for relation");
  ASSERT(snap_restore(path, relation, bytes, 0) == SNAP_ERR_OK, "snap_read failed");
#else
  int f = open(path, O_RDONLY);
  ASSERT(f >= 0, strerror(errno));
  struct stat stats;
  ASSERT(fstat(f, &stats) >= 0, strerror(errno));
  off_t bytes = stats.st_size;
  relation = (int64 *) malloc(bytes);
  ASSERT(relation, "failed to allocate memory for relation");
  ASSERT(read(f, relation, bytes) >= 0, strerror(errno));
  ASSERT(close(f) >= 0, strerror(errno));
#endif
  double finish = timer();
  printf("\t%f seconds for all I/O\n", finish - start);

  fields = 3;
  tuples = (bytes/sizeof(uint64))/fields;

  printf("\t%lu fields\n", fields);
  printf("\t%lu tuples\n", tuples);

  relInfo->tuples = tuples;
  relInfo->fields = fields;
  relInfo->relation = relation;
  return relInfo;
}


void printrelation(struct relationInfo *R) {
  printf("tuples, fields: %lu, %lu\n", R->tuples, R->fields);
  for (uint64 i = 0; i < R->tuples; i++) {
    for( uint64 j = 0; j < R->fields; j=j+1 ) {
      printf("\t%ld", R->relation[(i*R->fields)+j]);
    }
    printf("\n");
  }
}


RangeIter::RangeIter(uint64_t num, bool asEnd) 
  : num(num) {
    next = (asEnd)?num:0;
  }
    
uint64_t RangeIter::operator*() {
  return next;
}
    
RangeIter& RangeIter::operator++() {
  // TODO assert in bounds
  next++;
  return *this;
}
    
bool RangeIter::notequal(const RangeIter& o) const {
  return (this->next!=o.next) || (this->num!=o.num);
}


bool operator!=(const RangeIter& o1, const RangeIter& o2) {
  return o1.notequal(o2);
}

bool operator==(const RangeIter& o1, const RangeIter& o2) {
  return !(o1 != o2);
}
    

RangeIterable::RangeIterable(uint64_t num) : num(num) {}

RangeIter RangeIterable::begin() {
  return RangeIter(num);
}
RangeIter RangeIterable::end() {
  return RangeIter(num, true);
}

void write_count(const char* path, uint64_t count) {
  std::ofstream o;
  o.open(path);
  o << count << std::endl;
  o.close();
}
