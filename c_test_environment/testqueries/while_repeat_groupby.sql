select SUM(I.a) as a, I.c as b, SUM(I.b) as c from
  (select SUM(T3.a) as a, T3.c as b, SUM(T3.b) as c from T3 group by T3.c) as I
  group by I.c;
