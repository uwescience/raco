--unionall
select a, b from T2
union all
select r.a as a, t.b as b from R2 r, T2 t where r.b=t.a;
