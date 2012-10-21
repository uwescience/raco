#include "boolean.h"

using namespace std;

void BinaryExpression::PrintTo(ostream &os, int indent) {
  os <<
};

// AND, OR
class BinaryBooleanExpression : public BooleanExpression {
  public:
    BinaryBooleanExpression(BooleanExpression &left, BooleanExpression &right);
};

// attribute reference, literal
class Value {};

// =, !=, <, >, <=, >=
class BinaryBooleanOperator : public BooleanExpression {
  public:
    BinaryBooleanOperator(const Value &left, const Value &right) : left(left), right(right) {};
  protected:
    const Value &left;
    const Value &right;
};

template<typename T>
class Literal : public Value {
  public:
    Literal(T val) : value(val) {};
  protected:
    T value;
};

class Attribute : public Value {
  public:
    Attribute(string val) : value(val) {};
  protected:
    string &value;
};

class EQ : public BinaryBooleanOperator {
  public:
    EQ(const Value &left, const Value &right) : BinaryBooleanOperator(left, right) {};
};

