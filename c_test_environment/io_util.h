#ifndef _IO_UTIL_H
#define _IO_UTIL_H

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
struct relationInfo {
  uint64 tuples;
  uint64 fields;
  uint64 *relation;
};

struct relationInfo *inhale(char *path, struct relationInfo *relInfo);
struct relationInfo *binary_inhale(char *path, struct relationInfo *relInfo);

void printrelation(struct relationInfo *R);

#define ZAPPA

#endif // IO_UTIL_H
