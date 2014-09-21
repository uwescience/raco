#pragma once
#include <unordered_map>
#include <vector>
#include <utility>


#include <iostream>
#include "utils.h"

template <typename K, typename V>
void insert(std::unordered_map<K, std::vector<V>* >& hash, K key, V val) {
  auto r = hash.find(key);
  if (r != hash.end()) {
    (r->second)->push_back(val);
  } else {
    // TODO can we use the iterator r to insert
    std::vector<V> * newvec = new std::vector<V>();
    newvec->push_back(val);
    hash[key] = newvec;
  }
}

template <typename K, typename V>
void SUM_insert(std::unordered_map<K, V >& hash, K key, V val) {
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[key];
  slot += val;
}

template <typename K1, typename K2, typename V>
void SUM_insert(std::unordered_map<std::pair<K1,K2>, V, pairhash >& hash, K1 key1, K2 key2, V val) {
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[std::pair<K1,K2>(key1, key2)];
  slot += val;
}

template <typename K, typename V>
void COUNT_insert(std::unordered_map<K, V>& hash, K key, V val) {
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[key];
  slot += 1;
}

template <typename K1, typename K2, typename V>
void COUNT_insert(std::unordered_map<std::pair<K1,K2>, V, pairhash >& hash, K1 key1, K2 key2, V val) {
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[std::pair<K1,K2>(key1, key2)];
  slot += 1;
}

static uint64_t MIN(uint64_t a, uint64_t b) {
  return (a<b) ? a : b;
}

static uint64_t MAX(uint64_t a, uint64_t b) {
  return (a>b) ? a : b;
}

template <typename K, typename V>
void MIN_insert(std::unordered_map<K, V >& hash, K key, V val) {
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[key];
  slot = MIN(slot, val);
}

template <typename K, typename V>
void MAX_insert(std::unordered_map<K, V >& hash, K key, V val) {
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[key];
  slot = MAX(slot, val);
}

template <typename K1, typename K2, typename V>
void MIN_insert(std::unordered_map<std::pair<K1,K2>, V, pairhash >& hash, K1 key1, K2 key2, V val) {
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[std::pair<K1,K2>(key1,key2)];
  slot = MIN(slot, val);
}

template <typename T, typename K1, typename K2, typename V>
void MAX_insert(std::unordered_map<std::pair<K1,K2>, V, pairhash >& hash, K1 key1, K2 key2, V val) {
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[std::pair<K1,K2>(key1,key2)];
  slot = MAX(slot, val);
}

// one key
template <typename V>
void SUM_insert(V& var, V val) {
  var += val;
}

// one key
template <typename V>
void COUNT_insert(V& var, V val) {
  var += 1;
}

// one key
template <typename V>
void MIN_insert(V& var, V val) {
  var = MIN(var, val);
}

// one key
template <typename V>
void MAX_insert(V& var, V val) {
  var = MAX(var, val);
}

template <typename T, typename K>
std::vector<T>& lookup(std::unordered_map<K, std::vector<T>* >& hash, K key) {
  static std::vector<T> emptyResult;

  auto r = hash.find(key);
  if (r != hash.end()) {
    return *(r->second);
  } else {
    return emptyResult;
  }
}

