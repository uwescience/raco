#include "utils.h"

uint64_t identity_hash( int64_t k ) {
  return k;
}

uint64_t linear_hash( int64_t k) {
  return (73251599 * k + 110802387) % 98764321261;
}


// adapters for hash functions
static std::hash<int64_t> sh; 
uint64_t std_hash( int64_t k ) {
  return sh(k);
}

static pairhash ph;
uint64_t pair_hash( std::pair<int64_t, int64_t> k ) {
  return ph.operator()<int64_t, int64_t>(k);
}
