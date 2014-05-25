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
void SUM_insert(std::unordered_map<int64_t, int64_t >& hash, T tuple, uint64_t keypos, uint64_t valpos) {
  auto key = tuple.get(keypos);
  auto val = tuple.get(valpos);
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[key];
  slot += val;
}

template <typename T>
void COUNT_insert(std::unordered_map<int64_t, int64_t >& hash, T tuple, uint64_t keypos, uint64_t valpos) {
  auto key = tuple.get(keypos);
  auto val = tuple.get(valpos);
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[key];
  slot += 1;
}

// one key
template <typename T>
void SUM_insert(int64_t& var, T tuple, uint64_t valpos) {
  auto val = tuple.get(valpos);
  var += val;
}

// one key
template <typename T>
void COUNT_insert(int64_t& var, T tuple, uint64_t valpos) {
  var += 1;
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

