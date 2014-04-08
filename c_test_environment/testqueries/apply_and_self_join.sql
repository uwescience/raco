select X.a, Y.b from (select a as a, c as b from T3 where b < 4) X,
                     (select a as a, c as b from T3 where b < 4) Y
where X.b=Y.a;
