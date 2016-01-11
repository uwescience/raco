select SUM(R2.a), R2.b from R2, S2, T2 where R2.b=S2.a and S2.a=T2.a group by R2.b;
