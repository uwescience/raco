#pragma once

#include <functional>
#include <cstdint>
#include <utility>
#include <tuple>
#include <cstring>
#include <array>

uint64_t identity_hash( int64_t k );

uint64_t linear_hash( int64_t k);

uint64_t pair_hash( std::pair<int64_t, int64_t> k );

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

namespace std {
  template<size_t N>
  struct hash<std::array<char, N>>
  {
      typedef std::array<char, N> argument_type;
      typedef std::size_t result_type;

      result_type operator()(argument_type const& s) const {
        // use the std::hash for std::string
        return std::hash<std::string>()(std::string(s.data()));
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


namespace TupleUtils {

namespace impl {
  template <typename T, typename Sch, int total, int dst_o, int src_o>
    struct AssignmentHelper {
      void operator()(void ** x, T t) const {
        constexpr int i = total - dst_o;
        // convert to data type
        typename std::tuple_element<i,Sch>::type __dat = std::get<i>(t);
        // now store
        std::memcpy(&x[i], &__dat, sizeof(int64_t));
        AssignmentHelper<T, Sch, total, dst_o-1, src_o-1>()(x, t);
      }
    };

  template <typename T, typename Sch, int total, int dst_o>
    struct AssignmentHelper<T, Sch, total, dst_o, 0> {
      void operator()(void ** x, T t) const {
        return; //done
      }
    };
  
  template <typename T, int total, int i>
    struct StrHelper {
      void operator()(std::ostream& o, void ** x, T t) const {
        o << *((typename std::tuple_element<total-i, T>::type *) &x[total-i]) << ",";
        StrHelper<T, total, i-1>()(o, x, t);
      }
    };

  template <typename T, int total>
    struct StrHelper<T, total, 0> {
      void operator()(std::ostream& o, void ** x, T t) const {
        return; //done
      }
    };
}

  template <int dst_o, typename Sch, typename T>
  void assign(void ** x, T t) {
    constexpr size_t n = std::tuple_size<T>::value;
    impl::AssignmentHelper<T, Sch, n, dst_o+n, n>()(x, t);
  }
  
  template <typename T>
  std::ostream& str(std::ostream& o, void ** x, T t) {
    constexpr size_t n = std::tuple_size<T>::value;
    impl::StrHelper<T, n, n>()(o, x, t);
    return o;
  }
};
