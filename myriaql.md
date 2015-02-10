
MyriaQL  (pronounced "Miracle")
============================

MyriaQL is an imperative-yet-declarative high-level data flow language based on the relational algebra that includes support for iteration, user-defined functions, multiple expression languages, and familiar language constructs such as set comprehensions.  The language is the flagship programming interface for the Myria system, and can be compiled to a number of backends.

MyriaQL was designed by the Database group at the University of Washington, led by Andrew Whitaker, now at Amazon.

The language began as a ``whiteboard language'' for reasoning about the semantics of Datalog programs.  At the time, we anticipated Datalog becoming our premier programming interface.  But the fact that we were using an imperative style language to reason about Datalog made us realize we should just implement the imperative language directly.

MyriaQL is imperative: Each program is a sequence of assignment statements.  However, it is also declarative, in two ways: First, the optimizer is free to reorder blocks of code and apply other transformations as needed prior to execution, meaning that the programmer need not write the ``perfect'' program for decent performance.  Second, the right-hand-side of each assignment statement may itself be a declarative expression; programs may mix and match SQL and set comprehensions, for example. We find this combination of features to strike a useful balance between programmer control and programmer convenience.

Literate Example
================

```
-- this is a comment

-- Scan a relation
T1 = scan(TwitterK);

-- Find records where the first column = "foo bar"
T2 = [from T1 emit $0 == "foo bar" as x];

-- INCORRECT: No single-quoted literals
-- T2 = [from T1 emit $0 == 'foo bar' as x];

-- User-defined function
def triangleArea(a,b): (a*b)/2;
R = [from Foo emit triangleArea(x,y) as area];

-- User-defined aggregate function
apply RunningMean(value) {
      -- initialize the custom state, set cnt = 0 and summ = 0
      [0 as cnt, 0 as summ];
      -- for each record, add one one to the count (cnt) and add the record value to the sum (summ)
      [cnt + 1 as cnt, summ + value as summ];
      -- on each record, produce the running sum divided by the running count
      s / c;
};

-- A constant as a singleton relation
N = [2];

-- Create an empty relation with a particular schema
newBad = empty(id:int, v:float);

-- SQL-style wild cards
bc = [from emp emit emp.*];

out = [from emp where $0 * 2 == $1 emit *];
out = [from emp where $0 // $1 <> $1 emit *];

-- Unicode math operators ≤, ≥, ≠
out = [from emp where $0 ≤ $1 and $0 ≠ $1 and $1 ≥ $0 emit *];

-- Iteration
do
    mean = [from Good emit avg(v) as val];
    -- foo bar
    NewBad = [from Good where abs(Good.v - *mean) > *N * *std emit *];
    continue = diff(Good, NewBad);
while continue;

-- Save the result for future use
store(Good, OUTPUT);

-- other math functions
T3 = [from T1 emit sin(a)/4 + b as x];

-- typecast salary to a string using python-style function syntax
Groups = [FROM Emp EMIT id + 3, string(salary)];
```

MyriaQL for the SQL programmer
==============================

Join
----

```sql
SELECT E.name AS emp_name, D.name AS dept_name
  FROM public:adhoc:departments as D,  public:adhoc:employee as E
WHERE E.dept_id == D.id
  AND E.salary > 5000
```

```
out = [FROM SCAN(public:adhoc:departments) AS D, SCAN(public:adhoc:employee) AS E
       WHERE E.dept_id == D.id AND E.salary > 5000
       EMIT E.name AS emp_name, D.name AS dept_name];
STORE(out, OUTPUT);
```

Group By
-------

```sql
SELECT Emp.id, COUNT(salary) FROM Emp;
```

```
Emp = SCAN(public:adhoc:employee);
Groups = [FROM Emp EMIT COUNT(salary), Emp.id];
Store(Groups, OUTPUT, [$1]);
```

Advanced Examples
=================

* [PageRank in MyriaQL](https://github.com/uwescience/raco/blob/master/examples/pagerank.myl)
* [K-Means in MyriaQL](https://github.com/uwescience/raco/blob/master/examples/kmeans.myl)
* [Sigma Clipping in MyriaQL](https://github.com/uwescience/raco/blob/master/examples/sigma-clipping.myl)
