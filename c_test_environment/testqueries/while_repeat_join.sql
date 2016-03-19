-- iteration 2
select s1.b, s1.c, s1.a
    -- iteration 1
    from (select s1.b as a, s1.c as b, s1.a as c from T3 s1, T3 s2
    where s1.a=s2.b) as s1,
         (select s1.b as a, s1.c as b, s1.a as c from T3 s1, T3 s2
    where s1.a=s2.b) as s2
    where s1.a=s2.b;
