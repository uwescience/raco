/*
A(A,B,C) :- R(A, x, B), R(A, y, C), R(C, z, B)
*/
/*
Project(col0,col2,col3)[Select(col0 = col6)[Join(col3 = col2)[Join(col2 = col2)[Scan(R), Scan(R)], Scan(R)]]]
*/
//-------
/*
FilteringNLJoinChain([col2 = col2, col3 = col2], [1 = 1, 1 = 1], [1 = 1, 1 = 1], col0 = col6)[FileScan(R),FileScan(R),FileScan(R)]
*/
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


inline uint64 string_lookup(const char *key) {
  return 1; // dummy for now. 
}

inline bool equals(struct relationInfo *left, uint64 leftrow, uint64 leftattribute
                    , struct relationInfo *right, uint64 rightrow, uint64 rightattribute) {
  /* Convenience function for evaluating equi-join conditions */
  uint64 leftval = left->relation[leftrow*left->fields + leftattribute];
  uint64 rightval = right->relation[rightrow*right->fields + rightattribute];
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

  
// Compiled subplan for FilteringNLJoinChain([col2 = col2, col3 = col2], [1 = 1, 1 = 1], [1 = 1, 1 = 1], col0 = col6)[FileScan(R),FileScan(R),FileScan(R)]
printf("Evaluating subplan FilteringNLJoinChain([col2 = col2, col3 = col2], [1 = 1, 1 = 1], [1 = 1, 1 = 1], col0 = col6)[FileScan(R),FileScan(R),FileScan(R)]\n");

// Compiled subplan for FileScan(R)
printf("Evaluating subplan FileScan(R)\n");
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
// originalterm=Term_R(A,y,C) (position 1)
// originalterm=Term_R(A,x,B) (position 0)
// originalterm=Term_R(C,z,B) (position 2)
// symbol=V1

// Compiled subplan for FileScan(R)
printf("Evaluating subplan FileScan(R)\n");

struct relationInfo *V2;
V2 = V1;

// Compiled subplan for FileScan(R)
printf("Evaluating subplan FileScan(R)\n");

struct relationInfo *V3;
V3 = V1;


{ /* Begin Join Chain */

  printf("Begin Join Chain ['V1', 'V2', 'V3']\n");
  #pragma mta trace "running join 0"

  double start = timer();

  getCounters(counters, currCounter);
  currCounter = currCounter + 1; // 1

  // Loop over left leaf relation 
  for (uint64 V1_row = 0; V1_row < V1->tuples; V1_row++) {

      
      { /* Begin Join Level 0 */
       
        #pragma mta trace "running join 0"
      
        if (( (1) == (1) )) { // filter on join0.left 
          // Join 0
          for (uint64 V2_row = 0; V2_row < V2->tuples; V2_row++) {
            if (( (1) == (1) )) { // filter on join0.right          
              if (equals(V1, V1_row, 2 // left attribute ref
                       , V2, V2_row, 2)) { //right attribtue ref
      
                             
                  { /* Begin Join Level 1 */
                   
                    #pragma mta trace "running join 1"
                  
                    if (( (1) == (1) )) { // filter on join1.left 
                      // Join 1
                      for (uint64 V3_row = 0; V3_row < V3->tuples; V3_row++) {
                        if (( (1) == (1) )) { // filter on join1.right          
                          if (equals(V2, V2_row, 0 // left attribute ref
                                   , V3, V3_row, 2)) { //right attribtue ref
                  
                                         
                              if (( (V1->relation[V1_row*V1->fields + 0]) == (V3->relation[V3_row*V3->fields + 0]) )) {
                              
                                // Here we would send the tuple to the client, or write to disk, or fill a data structure
                                printf("joined tuple: %d, %d, %d\n", V1_row, V2_row, V3_row);
                                resultcount++;
                              
                              }
                  
                          } // Join 1 condition
                        } // filter on join1.right
                      } // loop over join1.right
                    } // filter on join1.left 
                  
                  }
                  
      
              } // Join 0 condition
            } // filter on join0.right
          } // loop over join0.right
        } // filter on join0.left 
      
      }
      

  } // loop over join0.left

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
  //free(resultInfo.relation);
}

