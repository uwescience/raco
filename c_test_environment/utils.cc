#include "utils.h"

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
