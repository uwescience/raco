// Precount_select: Use buckets to track the number of matches
// Use buckets to copy into the result array
#include <cstdio>
#include <cstdlib>     // for exit()
#include <fcntl.h>      // for open()
#include <unistd.h>     // for close()
#include <sys/stat.h>   // for fstat()
#include <ctype.h>      // for isdigit()
#include <cstring>
#include <errno.h>
#include <algorithm>
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

#include <iomanip>
#include <cstdint>
#include <iostream>
#include <fstream>
typedef int64_t int64;
typedef uint64_t uint64;

#include <unordered_map>
#include <vector>
#include <limits>
#endif

#include "io_util.h"
#include "hash.h"
#include "radish_utils.h"
#include "strings.h"
#include "timing.h"

// ------------------------------------------------------------------


{{declarations}}

StringIndex string_index;
void init( ) {
}


void query(struct relationInfo *resultInfo)
{
  printf("\nstarting Query stdout\n");fflush(stdout);

  double start = timer();

  uint64 resultcount = 0;
  struct relationInfo {{resultsym}}_val;
  struct relationInfo *{{resultsym}} = &{{resultsym}}_val;


  // -----------------------------------------------------------
  // Fill in query here
  // -----------------------------------------------------------
  {{initialized}}


 {{queryexec}}

  {{cleanups}}

  // return final result
  resultInfo->tuples = {{resultsym}}->tuples;
  resultInfo->fields = {{resultsym}}->fields;
  resultInfo->relation = {{resultsym}}->relation;

}



int main(int argc, char **argv) {

  struct relationInfo resultInfo;

  init();

    printf("post-init stdout\n");fflush(stdout);

  // Execute the query
  query(&resultInfo);

    printf("post-query stdout\n");fflush(stdout);

#ifdef ZAPPA
//  printrelation(&resultInfo);
#endif
//  free(resultInfo.relation);

    printf("exiting stdout\n");fflush(stdout);

}
