#pragma once

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

int * mallocCounterMemory(int numCounters);
void freeCounterMemory(int *counters);
void getCounters(int *counters, int currCounter);
void printDiffCounters(int *counters, int sz);
void printCounters(int *counters, int sz); 

