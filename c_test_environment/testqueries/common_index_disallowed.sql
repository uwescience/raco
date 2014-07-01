select t.a, t.b, r1.b, r2.a from T2 t, R2 r1, R2 r2
where t.a=r1.a
and r1.a=r2.b;
