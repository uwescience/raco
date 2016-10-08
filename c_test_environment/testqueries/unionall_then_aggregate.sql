select SUM(A2.a), A2.b from 
(select * from R2
 union all
 select * from S2) A2
group by A2.b;
