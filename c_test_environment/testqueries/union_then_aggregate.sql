select SUM(A1.a), A2.a from 
(select * from R2) A1,
union all
(select * from S2) A2
group by A2.a;
