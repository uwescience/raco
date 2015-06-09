select t1.a as src, t2.b as dst, count(t1.a) from T2 t1, T2 t2
where t1.b = t2.a
group by t1.a, t2.b;
