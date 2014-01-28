import test_query
import sys

if __name__ == "__main__":
  query = sys.argv[1]
  print query
  name = sys.argv[2]
  print name

  fname = test_query.testEmit(query, name, test_query.GrappaAlgebra)
