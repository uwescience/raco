#pragma once

#include <functional>

uint64_t identity_hash( int64_t k ) {
  return k;
}

uint64_t linear_hash( int64_t k) {
  return (73251599 * k + 110802387) % 98764321261;
}

static std::hash<int64_t> sh; 
uint64_t std_hash( int64_t k ) {
  return sh(k);
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

template <typename D, typename S>
D transpose(S s) {
  D d;
  for (int i=0; i<s.numFields(); i++) {
    d.set(i, s.get(i));
  }
  return d;
}
