import test_query
import sys

if __name__ == "__main__":
  queryfile = sys.argv[1]
  tr = 'sp2bench_1m'
  with open(queryfile, 'r') as f:
      query = f.read() % locals()

  fname = test_query.testEmit(query, queryfile, test_query.CCAlgebra)
  
  
