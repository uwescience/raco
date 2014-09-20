#pragma once

#include <functional>
#include <cstdint>
#include <utility>
#include <tuple>

uint64_t identity_hash( int64_t k );

// adapters for hash functions
template <typename T>
uint64_t std_hash( T k ) {
  return std::hash<T>()(k);
}

uint64_t linear_hash( int64_t k);

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


// hashing for tuples
// http://stackoverflow.com/questions/7110301/generic-hash-for-tuples-in-unordered-map-unordered-set
namespace hash_tuple{

  template <typename TT>
    struct hash
    {
      size_t
        operator()(TT const& tt) const
        {                                              
          return std::hash<TT>()(tt);                                 
        }                                              
    };
}


namespace hash_tuple{

  namespace impl
  {
    template <class T>
      inline void hash_combine(std::size_t& seed, T const& v)
      {
        seed ^= hash_tuple::hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
      }
  }
}

namespace hash_tuple{

  namespace impl
  {

    // Recursive template code derived from Matthieu M.
    template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
      struct HashValueImpl
      {
        static void apply(size_t& seed, Tuple const& tuple)
        {
          HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
          hash_combine(seed, std::get<Index>(tuple));
        }
      };

    template <class Tuple>
      struct HashValueImpl<Tuple,0>
      {
        static void apply(size_t& seed, Tuple const& tuple)
        {
          hash_combine(seed, std::get<0>(tuple));
        }
      };
  }

  template <typename ... TT>
    struct hash<std::tuple<TT...>> 
    {
      size_t
        operator()(std::tuple<TT...> const& tt) const
        {                                              
          size_t seed = 0;                             
          impl::HashValueImpl<std::tuple<TT...> >::apply(seed, tt);    
          return seed;                                 
        }                                              
    };

}

// adapter for hash function
template <typename T>
uint64_t hash_tuple_hash(T k) {
  return hash_tuple::hash<T>()(k);
}

