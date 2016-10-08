--unionall
select X.a, Y.a, Y.b from (select t.a as a, t.b as b from T2 t, R1 r where t.b < 4  and t.a=r.a
                      union all
                      select r.a as a, r.b as b from R2 r, T1 t where r.a=t.a) X,
                     (select t.a as a, t.b as b from T2 t, R1 r where t.b < 4  and t.a=r.a
                      union all
                      select r.a as a, r.b as b from R2 r, T1 t where r.a=t.a) Y
where X.b=Y.a;
