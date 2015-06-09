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

// How to use the counters:
// 1) malloc space for as many counters as you need
//    int numCounters = 7;
//    int *counters = mallocCounterMemory(numCounters);
// 2) Call getCounters(), specifying the particular address for storing the
//    counter values. Increment currCounter so that you don't overwrite values.
//    int currCounter = 0;
//    getCounters(counters, currCounter);
//    currCounter = currCounter + 1; // 1
// 3) Print the difference between the counters
//    printDiffCounters(counters, numCounters);
// 4) Free the memory storing the counter values
//    free(counters);
   



// --------------------------------- COUNTERS -------------------------------------
#define RT_ISSUES_IDX (0)
#define RT_MEMREFS_IDX (1)
#define RT_STREAMS_IDX (2)
#define RT_CONCURRENCY_IDX (3)
#define RT_CLOCK_IDX (4)
#define RT_PHANTOM_IDX (5)
#define RT_READY_IDX (6)
#define RT_TRAP_IDX (7)
#define NUM_RT_COUNTERS (RT_TRAP_IDX+1)

int *
mallocCounterMemory(int numCounters) {
#ifdef __MTA__
  int *counters = (int *)malloc(numCounters*(sizeof(int)*NUM_RT_COUNTERS));
  return counters;
#else // __MTA__
  return NULL;
#endif
}

void
freeCounterMemory(int *counters) {
#ifdef __MTA__
  free(counters);
#endif // __MTA__
}

bool countersInitialized = false;
int TrapCounter = -1;

void
initCounters()
{
#ifdef __MTA__
  TrapCounter = mta_reserve_task_event_counter(1, CNT_TRAP);
#endif // __MTA__
  countersInitialized = true;
}

void
getCounters(int *counters, int currCounter)
{
  if( countersInitialized == false ) {
    initCounters();
  }
#ifdef __MTA__
  int *res = &counters[currCounter*NUM_RT_COUNTERS];
  res[RT_ISSUES_IDX] = mta_get_task_counter(RT_ISSUES);
  res[RT_MEMREFS_IDX] = mta_get_task_counter(RT_MEMREFS);
  res[RT_STREAMS_IDX] = mta_get_task_counter(RT_STREAMS);
  res[RT_CONCURRENCY_IDX] = mta_get_task_counter(RT_CONCURRENCY);
  // Using RT_CLOCK gives some funky behavior that Simon might understand, but I don't.
  //res[RT_CLOCK_IDX] = mta_get_task_counter(RT_CLOCK);
  res[RT_CLOCK_IDX] = mta_get_clock(0);
  res[RT_PHANTOM_IDX] = mta_get_task_counter(RT_PHANTOM);
  res[RT_READY_IDX] = mta_get_task_counter(RT_READY);
  res[RT_TRAP_IDX] = mta_get_task_counter(RT_TRAP);
  //mta_report_trap_counters();
#endif // __MTA__
}

void
printCounters(int *counters, int sz) 
{
#ifdef __MTA__
  int i;
  for(i=0; i < sz; i=i+1 ) {
    int *res =  &counters[i*NUM_RT_COUNTERS];
    printf("counter[%d][RT_ISSUES] = %d\n", i, res[RT_ISSUES_IDX]);
    printf("counter[%d][RT_MEMREFS] = %d\n", i, res[RT_MEMREFS_IDX]);
    printf("counter[%d][RT_STREAMS] = %d\n", i, res[RT_STREAMS_IDX]);
    printf("counter[%d][RT_CONCURRENCY] = %d\n", i, res[RT_CONCURRENCY_IDX]);
    printf("counter[%d][RT_CLOCK] = %d\n", i, res[RT_CLOCK_IDX]);
    printf("counter[%d][RT_PHANTOM] = %d\n", i, res[RT_PHANTOM_IDX]);
    printf("counter[%d][RT_READY] = %d\n", i, res[RT_READY_IDX]);
    printf("counter[%d][RT_TRAP] = %d\n", i, res[RT_TRAP_IDX]);
  }
#endif // __MTA__
}

void
printDiffCounters(int *counters, int sz) 
{
#ifdef __MTA__
  int i;
  double freq = mta_clock_freq();

  printf("DIFFS:\t\tCLOCK\t\tISSUES\t\tMEMREFS\t\tPHANTOM\t\tSTREAMS\t\tCONCURRENCY\tREADY\tTRAP\n");

  for(i=1; i < sz; i=i+1 ) {
    int *res =  &counters[i*NUM_RT_COUNTERS];
    int *prevres = &counters[(i-1)*NUM_RT_COUNTERS];
    int diff_issues = (res[RT_ISSUES_IDX]-prevres[RT_ISSUES_IDX]);
    int diff_memrefs = (res[RT_MEMREFS_IDX]-prevres[RT_MEMREFS_IDX]);
    int diff_streams = 256 * (res[RT_STREAMS_IDX]-prevres[RT_STREAMS_IDX]);
    int diff_concurrency = 256 * (res[RT_CONCURRENCY_IDX]-prevres[RT_CONCURRENCY_IDX]);
    int diff_clock = (res[RT_CLOCK_IDX]-prevres[RT_CLOCK_IDX]) ;
    double diff_clock_secs = (res[RT_CLOCK_IDX]-prevres[RT_CLOCK_IDX]) / freq;
    int diff_phantom = (res[RT_PHANTOM_IDX]-prevres[RT_PHANTOM_IDX]);
    int diff_ready = (res[RT_READY_IDX]-prevres[RT_READY_IDX]); 
    int diff_trap = (res[RT_TRAP_IDX]-prevres[RT_TRAP_IDX]); 


    printf("DIFF[%d-%d][RT_ISSUES] = %d\t\tcounter[%d][RT_ISSUES] = %d\tcounter[%d][RT_ISSUES] = %d\n", 
	   i, i-1, diff_issues,
	   i-1, prevres[RT_ISSUES_IDX], i, res[RT_ISSUES_IDX]); 
    printf("DIFF[%d-%d][RT_MEMREFS] = %d\t\tcounter[%d][RT_MEMREFS] = %d\tcounter[%d][RT_MEMREFS] = %d\n", 
	   i, i-1, diff_memrefs,
	   i-1, prevres[RT_MEMREFS_IDX], i, res[RT_MEMREFS_IDX]); 
    printf("DIFF[%d-%d][RT_STREAMS] = %d\t\tcounter[%d][RT_STREAMS] = %d\tcounter[%d][RT_STREAMS] = %d\n",
	   i, i-1, diff_streams,
	   i-1, prevres[RT_STREAMS_IDX], i, res[RT_STREAMS_IDX]); 
    printf("DIFF[%d-%d][RT_CONCURRENCY] = %d\t\tcounter[%d][RT_CONCURRENCY] = %d\tcounter[%d][RT_CONCURRENCY] = %d\n", 
	   i, i-1, diff_concurrency,
	   i-1, prevres[RT_CONCURRENCY_IDX], i, res[RT_CONCURRENCY_IDX]); 
    printf("DIFF[%d-%d][CLOCK] = %d (%f secs)\t\tcounter[%d][RT_CLOCK] = %d\tcounter[%d][RT_CLOCK] = %d\n", 
	   i, i-1, diff_clock, diff_clock_secs,
	   i-1, prevres[RT_CLOCK_IDX], i, res[RT_CLOCK_IDX]); 
    printf("DIFF[%d-%d][RT_PHANTOM] = %d\t\tcounter[%d][RT_PHANTOM] = %d\tcounter[%d][RT_PHANTOM] = %d\n", 
	   i, i-1, diff_phantom,
	   i-1, prevres[RT_PHANTOM_IDX], i, res[RT_PHANTOM_IDX]); 
    printf("DIFF[%d-%d][RT_READY] = %d\t\tcounter[%d][RT_READY] = %d\tcounter[%d][RT_READY] = %d\n",
	   i, i-1, diff_ready,
	   i-1, prevres[RT_READY_IDX], i, res[RT_READY_IDX]);
    printf("DIFF[%d-%d][RT_TRAP] = %d\t\tcounter[%d][RT_TRAP] = %d\tcounter[%d][RT_TRAP] = %d\n",
	   i, i-1, diff_trap,
	   i-1, prevres[RT_TRAP_IDX], i, res[RT_TRAP_IDX]);


    printf("DIFFS[%d-%d]:\t%d (%f)\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n\n", i, i-1,
	   diff_clock, diff_clock_secs, diff_issues, diff_memrefs, diff_phantom, diff_streams, diff_concurrency, diff_ready, diff_trap);
  }
#endif // __MTA__
}

// --------------------------------- COUNTERS -------------------------------------
