#pragma once

#include <functional>
#include <cstdint>
#include <utility>

uint64_t identity_hash( int64_t k );

uint64_t linear_hash( int64_t k);

uint64_t std_hash( int64_t k );

uint64_t pair_hash( std::pair<int64_t, int64_t> k );


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

class pairhash {
  public:
  template <typename T, typename U>
    std::size_t operator()(const std::pair<T, U> &x) const {
      auto ha = std::hash<T>()(x.first);
      auto hb = std::hash<U>()(x.second);
      // h(a) * (2^32 + 1) + h(b)
      return (ha << 32) + ha + hb;
    }
};

namespace std {
  template<>
    struct hash<std::pair<int64_t,int64_t>>
    {
      typedef std::pair<int64_t,int64_t> argument_type;
      typedef std::size_t result_type;

      result_type operator()(argument_type const& s) const {
        result_type const h1 ( std::hash<int64_t>()(s.first) );
        result_type const h2 ( std::hash<int64_t>()(s.second) );
        return h1 ^ (h2 << 1);
      }
    };
}
