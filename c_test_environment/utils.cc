#include "utils.h"

uint64_t identity_hash( int64_t k ) {
  return k;
}

uint64_t linear_hash( int64_t k) {
  return (73251599 * k + 110802387) % 98764321261;
}

static pairhash ph;
uint64_t pair_hash( std::pair<int64_t, int64_t> k ) {
  return ph.operator()<int64_t, int64_t>(k);
}
