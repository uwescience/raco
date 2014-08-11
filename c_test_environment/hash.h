#pragma once
#include <unordered_map>
#include <vector>
#include <utility>


#include <iostream>

template <typename T, typename K>
void insert(std::unordered_map<K, std::vector<T>* >& hash, T tuple, uint64_t pos) {
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

struct pairhash {
  template <typename T, typename U>
    std::size_t operator()(const std::pair<T, U> &x) const {
      auto ha = std::hash<T>()(x.first);
      auto hb = std::hash<U>()(x.second);
      // h(a) * (2^32 + 1) + h(b)
      return (ha << 32) + ha + hb;
    }
};

template <typename T, typename K, typename V>
void SUM_insert(std::unordered_map<K, V >& hash, T tuple, uint64_t keypos, uint64_t valpos) {
  auto key = tuple.get(keypos);
  auto val = tuple.get(valpos);
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[key];
  slot += val;
}

template <typename T, typename K1, typename K2, typename V>
void SUM_insert(std::unordered_map<std::pair<K1,K2>, V, pairhash >& hash, T tuple, uint64_t key1pos, uint64_t key2pos, uint64_t valpos) {
  auto key1 = tuple.get(key1pos);
  auto key2 = tuple.get(key2pos);
  auto val = tuple.get(valpos);
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[std::pair<K1,K2>(key1, key2)];
  slot += val;
}

template <typename T, typename K, typename V>
void COUNT_insert(std::unordered_map<K, V>& hash, T tuple, uint64_t keypos, uint64_t valpos) {
  auto key = tuple.get(keypos);
  auto val = tuple.get(valpos);
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[key];
  slot += 1;
}

template <typename T, typename K1, typename K2, typename V>
void COUNT_insert(std::unordered_map<std::pair<K1,K2>, V, pairhash >& hash, T tuple, uint64_t key1pos, uint64_t key2pos, uint64_t valpos) {
  auto key1 = tuple.get(key1pos);
  auto key2 = tuple.get(key2pos);
  auto val = tuple.get(valpos);
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

template <typename T, typename K, typename V>
void MIN_insert(std::unordered_map<K, V >& hash, T tuple, uint64_t keypos, uint64_t valpos) {
  auto key = tuple.get(keypos);
  auto val = tuple.get(valpos);
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[key];
  slot = MIN(slot, val);
}

template <typename T, typename K, typename V>
void MAX_insert(std::unordered_map<K, V >& hash, T tuple, uint64_t keypos, uint64_t valpos) {
  auto key = tuple.get(keypos);
  auto val = tuple.get(valpos);
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[key];
  slot = MAX(slot, val);
}

template <typename T, typename K1, typename K2, typename V>
void MIN_insert(std::unordered_map<std::pair<K1,K2>, V, pairhash >& hash, T tuple, uint64_t key1pos, uint64_t key2pos, uint64_t valpos) {
  auto key1 = tuple.get(key1pos);
  auto key2 = tuple.get(key2pos);
  auto val = tuple.get(valpos);
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[std::pair<K1,K2>(key1,key2)];
  slot = MIN(slot, val);
}

template <typename T, typename K1, typename K2, typename V>
void MAX_insert(std::unordered_map<std::pair<K1,K2>, V, pairhash >& hash, T tuple, uint64_t key1pos, uint64_t key2pos, uint64_t valpos) {
  auto key1 = tuple.get(key1pos);
  auto key2 = tuple.get(key2pos);
  auto val = tuple.get(valpos);
  // NOTE: this method is only valid for 0 identity functions
  auto& slot = hash[std::pair<K1,K2>(key1,key2)];
  slot = MAX(slot, val);
}

// one key
template <typename T, typename V>
void SUM_insert(V& var, T tuple, uint64_t valpos) {
  auto val = tuple.get(valpos);
  var += val;
}

// one key
template <typename T, typename V>
void COUNT_insert(V& var, T tuple, uint64_t valpos) {
  var += 1;
}

// one key
template <typename T, typename V>
void MIN_insert(V& var, T tuple, uint64_t valpos) {
  auto val = tuple.get(valpos);
  var = MIN(var, val);
}

// one key
template <typename T, typename V>
void MAX_insert(V& var, T tuple, uint64_t valpos) {
  auto val = tuple.get(valpos);
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

