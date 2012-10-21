#include "boolean.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {

  BooleanExpression cond = EQ(Attribute(string("X")), Literal<int>(1));
  
  cond.PrintTo(cond);
  //cout << cond << endl;
}

