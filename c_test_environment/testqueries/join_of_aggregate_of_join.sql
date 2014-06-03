select a.rsum, s.b from S2 s,
    (select SUM(r.a) as rsum, t.b as tc from R2 r, T2 t 
        where r.b = t.a group by t.b) a
where
a.tc = s.a;
