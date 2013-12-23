#pragma once
#include <unordered_map>
#include <vector>


#include <iostream>

template <typename T>
void insert(std::unordered_map<int64_t, std::vector<T>* >& hash, T tuple, uint64_t pos) {
  auto key = tuple.get(pos);
  auto r = hash.find(key);
  if (r != hash.end()) {
    (r->second)->push_back(tuple);
  } else {
    // TODO can we use the iterator r to insert
    std::vector<T> * newvec = new std::vector<T>();
    newvec->push_back(tuple);
    hash[key] = newvec;
  }
}

template <typename T>
std::vector<T>& lookup(std::unordered_map<int64_t, std::vector<T>* >& hash, int64_t key) {
  static std::vector<T> emptyResult;

  auto r = hash.find(key);
  if (r != hash.end()) {
    return *(r->second);
  } else {
    return emptyResult;
  }
}

template <typename D, typename S1, typename S2>
D combine(S1 s1, S2 s2) {
  D d;
  for (int i=0; i<s1.numFields(); i++) {
    d.set(i, s1.get(i));
  }
  for (int i=0; i<s2.numFields(); i++) {
    d.set(s1.numFields()+i, s2.get(i));
  }
  return d;
}
