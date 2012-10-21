// Precount_select: Use buckets to track the number of matches
// Use buckets to copy into the result array
#include <stdio.h>
#include <stdlib.h>     // for exit()
#include <fcntl.h>      // for open()
#include <unistd.h>     // for close()
#include <sys/stat.h>   // for fstat()
#include <ctype.h>      // for isdigit()
#include <string.h>
#include <errno.h>
#include <sys/types.h>
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
#include "counters_util.h"

// ------------------------------------------------------------------

#define Subject   0
#define Predicate 1
#define Object    2
#define Graph     3

#define XXX  330337405
#define YYY 1342785348
#define ZZZ 1395042699

#define buckets 100000

const uint64 mask = (1L << 53) - 1;
/*
// Insert a value into a hash table
void insert(uint64 **ht1, uint64 size1, uint64 offset)
{
  uint64 hash = (uint64(offset) & mask) % size1;
#ifdef __MTA__
  while (1) {
    if (!readff(ht1 + hash)) {
      uint64 *p = readfe(ht1 + hash); // lock it
      if (p) writeef(ht1 + hash, p); // unlock and try again
      else break;
    }
    hash++;
    if (hash == size1)
    hash = 0;
  }
  writeef(ht1 + hash, relation2 + i); // unlock it
#else
  while (ht1[hash]) {
    hash++;
    if (hash == size1) hash = 0;
  }
  ht1[hash] = relation2 + i;
#endif
}
*/

inline bool check_condition(struct relationInfo *left, struct relationInfo *right
                           , uint64 leftrow, uint64 rightrow
                           , uint64 leftattribute, uint64 rightattribute) {
  /* Convenience function for evaluating equi-join conditions */
  uint64 leftval = left->relation[leftrow*left->fields + leftattribute];
  uint64 rightval = right->relation[rightrow*right->fields + rightattribute];
  //printf("rows: %d, %d\n", leftrow, rightrow);
  //printf("checking: %d, %d\n", leftval, rightval);
  return leftval == rightval;
}


void query(struct relationInfo *resultInfo)
{
  printf("\nstarting Query\n");

  int numCounters = 7;
  int currCounter = 0;
  int *counters = mallocCounterMemory(numCounters);

  double start = timer();

  getCounters(counters, currCounter);
  currCounter = currCounter + 1; // 1
  
  uint64 resultcount = 0;
  struct relationInfo resultRelation_val;
  struct relationInfo *resultRelation = &resultRelation_val;


  // -----------------------------------------------------------
  // Fill in query here
  // -----------------------------------------------------------

  

/*
=====================================
  Scan(R)
=====================================
*/

printf("V1 = Scan(R)\n");

struct relationInfo V1_val;

#ifdef __MTA__
  binary_inhale("R", &V1_val);
  //inhale("R", &V1_val);
#else
  inhale("R", &V1_val);
#endif // __MTA__

struct relationInfo *V1 = &V1_val;
 
/*
=====================================
  Scan(R)
=====================================
*/

printf("V2 = Scan(R)\n");

struct relationInfo V2_val;

#ifdef __MTA__
  binary_inhale("R", &V2_val);
  //inhale("R", &V2_val);
#else
  inhale("R", &V2_val);
#endif // __MTA__

struct relationInfo *V2 = &V2_val;

{ // Begin Filtering_NestedLoop_Join_Chain



  printf("V2 = Join(V1,V1) \n");
  // Assume left-deep plan

  // leaves of the tree
  
struct relationInfo *rel1 = V1;
struct relationInfo *rel2 = V2;


  // Join 1
  
struct relationInfo *join1_left = rel1;
uint64 join1_leftattribute = 2;

struct relationInfo *join1_right = rel2;
uint64 join1_rightattribute = 0;


  double start = timer();

  getCounters(counters, currCounter);
  currCounter = currCounter + 1; // 1

#pragma mta trace "running join"
  // Left Root
  for (uint64 join1_leftrow = 0; join1_leftrow < join1_left->tuples; join1_leftrow++) {
    if (true) { // filter on join1.left
      // Join 1
      for (uint64 join1_rightrow = 0; join1_rightrow < join1_right->tuples; join1_rightrow++) {
        if (true) { // filter on join1.right
          if (check_condition(join1_left, join1_right
                             , join1_leftrow, join1_rightrow, join1_leftattribute, join1_rightattribute)) {
             


printf("joined tuple: %d, %d\n", join1_leftrow, join1_rightrow);
resultcount++;



          } // Join 1 condition
        } // filter on join1.right
      } // loop over join1.right
    } // filter on join1.left 
  } // loop over join1.left

} // End Filtering_NestedLoop_Join_Chain





  // return final result
  resultInfo->tuples = resultRelation->tuples;
  resultInfo->fields = resultRelation->fields;
  resultInfo->relation = resultRelation->relation;

}



int main(int argc, char **argv) {

  struct relationInfo resultInfo;

  // Execute the query
  query(&resultInfo);

#ifdef ZAPPA
  printrelation(&resultInfo);
#endif
  free(resultInfo.relation);
}

