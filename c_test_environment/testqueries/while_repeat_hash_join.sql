select s1.b, s1.c, s1.a
    from (select s1.b, s1.c, s1.a from T3 s1, T3 s2
    where s1.a=s2.b) as s1,
         (select s1.b, s1.c, s1.a from T3 s1, T3 s2
    where s1.a=s2.b) as s2
    where s1.a=s2.b;