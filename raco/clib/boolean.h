#include <iostream>

using namespace std;

template<class T>
class BinaryOperator {
  public:
    BinaryOperator(const T &left, const T& right) : left(left), right(right) {};
    virtual void PrintTo(ostream &os, int indent); 
  protected:
    const T &left;
    const T &right;
};

// all boolean expressions happen to be binary operators currently
template<class T>
class BooleanExpression : public BinaryOperator<T> {
    BooleanExpression(const T &left, const T& right) : BinaryOperator<T>(left, right) {};
};

// AND, OR
class BinaryBooleanExpression : public BooleanExpression<BooleanExpression> {
  public:
    BinaryBooleanExpression(BooleanExpression &left, BooleanExpression &right) : BooleanExpression<BooleanExpression>(left, right) {};
};

// attribute reference or literal
class Value {};

// =, !=, <, >, <=, >=
class Comparator : public BooleanExpression<Value> {
  public:
    Comparator(const Value &left, const Value &right) : BooleanExpression<Value>(left, right) {};
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

